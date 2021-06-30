import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import time
class bertclassifier(nn.Module):
    def __init__(self, bert, hid_dim, out_dim):
        super().__init__()
        self.bert = bert
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.relu = torch.nn.ReLU()
        self.fc_2 = nn.Linear(256, out_dim)

    def forward(self, src, src_mask, segment):
        # batch_size = src.shape[0]
        # src_len = src.shape[1]
        # pos = self.bert.pos_embedding(torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(src.device))
        # segment = self.bert.segment_embedding(segment)
        # src = self.bert.tok_embedding(src) + pos + segment

        # for layer in self.bert.layers:
        #     src = layer(src, src_mask)

        # self.bert.eval()
        src = self.bert.encode(src, src_mask, segment)
        out = self.relu(self.fc_1(src[:, 0, :]))
        out = self.fc_2(out)
        return out

def train(iterator, model, optimizer, loss_fn,  device):
    model.to(device)
    ret_loss = 0
    for batch_dict, label in iterator:
        optimizer.zero_grad()

        input_tokens = batch_dict["input_ids"].to(device)
        segments = batch_dict["token_type_ids"].to(device)
        attention_masks = batch_dict["attention_mask"].unsqueeze(1).unsqueeze(2).to(device)
        label = label.to(device)

        out = model(input_tokens, attention_masks, segments)

        # flatten_output = output.reshape(-1, output.size(-1))
        total_loss = loss_fn(out, label)
        total_loss.backward()
        ret_loss += total_loss.item()
        optimizer.step()

    epoch_loss = ret_loss / len(iterator)
    return epoch_loss

def evaluate_simple(iterator, model, loss_fn, device):
    model.eval()
    epoch_loss = 0
    model = model.to(device)
    acc = 0
    with torch.no_grad():
        step = 0
        f1 = 0
        for batch_dict, label in iterator:
            st = time.time()
            input_tokens = batch_dict["input_ids"].long().to(device)
            segments = batch_dict["token_type_ids"].long().to(device)
            attention_masks = batch_dict["attention_mask"].long().unsqueeze(1).unsqueeze(2).to(device)
            label = label.to(device)

            output = model(input_tokens, attention_masks, segments)
            loss = loss_fn(output, label)

            step_loss_val = loss.item()
            epoch_loss += step_loss_val

            ids = torch.argmax(output, dim=-1)
            mini_f1 = f1_score(label.reshape(-1).to('cpu'), ids.reshape(-1).to('cpu'), average='macro')
            hit = sum(list(label.reshape(-1).to('cpu') == ids.reshape(-1).to('cpu').flatten()))
            step += 1
            f1 += mini_f1
            acc += hit/len(list(label))
            # print(f"step_loss : {step_loss_val:.3f}, {step}/{len(iterator)}({step/len(iterator)*100:.2f}%) time : {time.time()-st:.3f}s", end="\r")

    return epoch_loss / len(iterator), acc/len(iterator)*100, f1/len(iterator)*100

if __name__ == "__main__":
    from data import *
    from torch.utils.data import DataLoader

    vocab_txt_path = "/home/jack/torchstudy/06Jun/BERT/vocabs/peti_namu_2021062409"
    txt_path = "/home/jack/torchstudy/05May/ELMo/data/ynat/train_tokenized.ynat"
    valid_txt_path = "/home/jack/torchstudy/05May/ELMo/data/ynat/val_tokenized.ynat"
    
    dataset = KoDataset_with_label_ynat(vocab_txt_path, txt_path)
    train_data_loader = DataLoader(dataset, collate_fn= dataset.collate_fn, batch_size=256)
    
    dataset = KoDataset_with_label_ynat(vocab_txt_path, valid_txt_path)
    valid_data_loader = DataLoader(dataset, collate_fn= dataset.collate_fn, batch_size=256)

    bert = torch.load('../models/best_petition_saved.pt').module
    classifier = bertclassifier(bert, 768, 7)

    optimizer = torch.optim.Adam(classifier.parameters(), lr = 0.000001, weight_decay = 0.01)
    loss_fn = torch.nn.CrossEntropyLoss()
    device = 'cuda:1'
    for i in range(100):
        train(train_data_loader, classifier, optimizer, loss_fn,  device)
        loss, acc, f1 = evaluate_simple(valid_data_loader, classifier, loss_fn,  device)
        print(f"{loss:.3f} / {acc:.3f} / {f1:.3f}")
