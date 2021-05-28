from src.trainer import *
from src.data import *
from src.infer import *
from src.model import *
import dill # for save
import os
import sys
import copy

def save_vocab(save_path, obj):
    with open(save_path, 'wb') as f:
        dill.dump(obj, f)

def read_vocab(save_path):
    with open(save_path, 'rb') as f:
        a = dill.load(f)
    return a

def train():
    
    BATCH_SIZE = 32

    train_data_paths = [
        "./data/src.tr",
        "./data/dst.tr"
    ]


    valid_data_paths = [
        "./data/src.valid",
        "./data/dst.valid"
    ]

    TrainDataset = KoKodataset(train_data_paths)
    TrainDataloader = DataLoader(TrainDataset, batch_size = BATCH_SIZE, shuffle=True, collate_fn=TrainDataset.batch_collate_fn)

    ValidDataset = KoKodataset(valid_data_paths)
    ValidDataset.src_vocab = TrainDataset.src_vocab
    ValidDataset.dst_vocab = TrainDataset.dst_vocab

    ValidDataloader = DataLoader(ValidDataset, batch_size = BATCH_SIZE, shuffle=True, collate_fn=ValidDataset.batch_collate_fn)

    configs = {
        'input_dim' : len(TrainDataset.src_vocab),
        'output_dim' : len(TrainDataset.dst_vocab),
        'src_pad_idx' : 0,
        'trg_pad_idx' : 0,
        'device' : 'cuda:1',
        'epochs' :  150,
    }

    trainer = Trainer(configs)
    trainer.run(TrainDataloader, ValidDataloader)
    return trainer, TrainDataset, ValidDataset, configs

if __name__ == '__main__':
    option = sys.argv[1]

    save_path = "./model"
    if option == "train":
        trainer, TrainDataset, ValidDataset, config = train()
        save_vocab(os.path.join(save_path, 'vocab/train_vocab.pkl'), copy.deepcopy(TrainDataset))

    elif option == "test":
        model_path = os.path.join(save_path, '3_best_model_36.pt') #159
        vocabs = read_vocab(os.path.join(save_path, 'vocab/train_vocab.pkl'))
        device = 'cpu'

        INPUT_DIM = 3004
        OUTPUT_DIM = 3004
        
        HID_DIM = 256
        ENC_LAYERS = 3
        DEC_LAYERS = 3
        ENC_HEADS = 8
        DEC_HEADS = 8
        ENC_PF_DIM = 512
        DEC_PF_DIM = 512
        ENC_DROPOUT = 0.1
        DEC_DROPOUT = 0.1

        enc = Encoder(INPUT_DIM, 
                    HID_DIM, 
                    ENC_LAYERS, 
                    ENC_HEADS, 
                    ENC_PF_DIM, 
                    ENC_DROPOUT, 
                    device)

        dec = Decoder(OUTPUT_DIM, 
                    HID_DIM, 
                    DEC_LAYERS, 
                    DEC_HEADS, 
                    DEC_PF_DIM, 
                    DEC_DROPOUT, 
                    device)

        model = Seq2Seq(enc, dec, 0, 0, device).to(device)
        model.load_state_dict(torch.load(model_path))
        print("---Load Finish")
        model.eval()
        while True:
            try:
                sentence = input("please type input \n -> ")
                src_indexes = vocabs.src_vocab.stoi(sentence, option='seq2seq')

                vocabs.dst_vocab.build_index_dict()
                trg_index_dict = vocabs.dst_vocab.index_dict
                
                trg_sos_index = 2
                trg_eos_index = 3
                translation_idx, attention = translate_sentence(src_indexes, trg_sos_index, trg_eos_index, model, 'cpu', max_len = 50)
                translation = " ".join([trg_index_dict[i] for i in translation_idx if i not in [0, 1, 2, 3]])
                print(f'predicted trg = {translation}')
            except Exception as e:
                print(e)