from models import *
from data import EnFrData
from trainer import Trainer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import math
import time
from functools import partial
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train_model_with_hp(config):
    dataObj = EnFrData(config) # device, batch_size
    
    config['input_dim'] = len(dataObj.SRC.vocab)
    config['output_dim'] = len(dataObj.TRG.vocab)
    config['pad_idx'] = dataObj.TRG.vocab.stoi[dataObj.TRG.pad_token] 
    trainer = Trainer(config) # input_dim, output_dim, emb_dim, hid_dim, emb_dim

    CLIP = 1
    trainer.init_weights()
    best_valid_loss = float('inf')

    for epoch in range(config['epochs']):
        
        start_time = time.time()
        
        train_loss, train_bleu = trainer.train(dataObj.train_iterator, CLIP)
        valid_loss, valid_bleu = trainer.evaluate(dataObj.valid_iterator, epoch, dataObj.TRG.vocab, f"emb{config['emb_dim']}_hid{config['hid_dim']}")
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(trainer.model.state_dict(), f"emb{config['emb_dim']}_hid{config['hid_dim']}.model.pt")
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Train BLEU: {train_bleu:7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f} | Val. BLEU: {valid_bleu:7.3f}')

if __name__ == "__main__":

    parameters = {
        "emb_dim" : 256,
        "hid_dim" : 512,
        "device" : torch.device('cuda:1'),
        "epochs" : 20,
        "batch_size" : 64
    }
    train_model_with_hp(parameters)
    
