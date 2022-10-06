import json as js
import string
import nltk 
import re
import numpy as np
import torch
import torch.nn as nn
import pickle
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from train import model_train
from model import Net

from  matplotlib import pyplot as plt

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu") #if gpu presesnt cuda:0 else cpu only
torch.backends.cudnn.benchmark = True


def nlp_preprocess(text): 
    global max_len
    clean = re.compile('<.*?>')
    ret = re.sub(clean,'',text) #Remove HTML tags
    ret = "".join([i for i in ret if i not in string.punctuation]) #Remove punctuations
    ret = ret.lower() #Lower the case of letters
    ret = ret.split()
    ret = [i for i in ret if i not in nltk.corpus.stopwords.words('english') and i!='not'] #Remove stopwords but keep not
    ret = " ".join(ret) 
    return ret

class CustomDataSet(Dataset):
    def __init__(self, data, embed):
        self.data = data
        self.embed = embed
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        
        #use word embedding(glove) to encode sentences 
        X = nlp_preprocess(self.data.iloc[idx]['review'])
        X = X.split()
        
        X = [self.embed[z] if z in self.embed else -1 for z in X] #The last embedding corresponds to UNK
        
        if self.data.iloc[idx]['sentiment'] == 'positive':
            lab = 1
        else:
            lab = 0

        sample = (X,lab)
        return sample



embeddings = None

#batch(input to this function) would be a list of tuples corresponding to the indices used by dataloader
def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    
    #pack the padded
    with torch.no_grad():
        X_l = []
        X_len = []
        for t in batch:
            tor = embeddings[t[0]]
            if tor.ndim ==2 and tor.shape[1] == dim:
                X_l.append(tor)
                X_len.append(tor.shape[0])
            else:
                print('stale index',t[2])
                X_l.append(embeddings[-1])
                X_len.append(1)
        X_seq = torch.nn.utils.rnn.pad_sequence(X_l,batch_first = True) #Generating padded sequences of variable length
        X_batch = torch.nn.utils.rnn.pack_padded_sequence(X_seq,X_len, batch_first = True, enforce_sorted=False)
        
        label_batch = torch.tensor([t[1] for t in batch])
        
    return X_batch,label_batch

criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':
    Macc = None
    Mdim = None
    model = None
    save_weights = {}
    final_embed = None
    for dim in [100, 200, 300]:
        #For creating a python dictionary between words and their glove embeddings(torch tensor form) 
        Path = '/raid/home/anikethv/glove/glove.6B/glove.6B.{}d.txt'.format(dim)
        i=0
        vec = []
        embedmap = dict()

        with open(Path,'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = [float(i) for i in values[1:]]
                if len(vector)!=dim:
                    vector = [float(0.1) for i in range(0,dim)]
                vec.append(vector)
                embedmap[word] = i
                i+=1
        vector = [float(0.1) for i in range(0,dim)] #Add another vector for unknown token(catchall for all Unknowns--UNK)
        vec.append(vector)
        embeddings = torch.tensor(vec, dtype=torch.float, requires_grad=True, device=device)
        tdata = CustomDataSet(pd.read_csv('Train dataset.csv'), embed=embedmap)
        print('Training start for Dimension : {}'.format(dim))
        model = Net(dim=dim).to(device)
        params = list(model.parameters())+[embeddings]
        optimiser = torch.optim.Adam(params, lr=0.001)
        vacc, epoch_vs_loss = model_train(model, train_data=tdata, criterion=criterion, optimiser=optimiser, verbose=True, 
                                    collate_fn=collate_fn_padd, embeddings=embeddings)
        print('\n\nAccuracy on Validation set:{}\n'.format(vacc))
        p = 'LSTM_{}.pt'.format(dim)
        torch.save(model.state_dict(), p)
        p = 'tuned_embed_{}.pt'.format(dim)
        torch.save(embeddings,p)
        if Macc is None or vacc > Macc:
            Macc = vacc
            Mdim = dim
            for name, param in model.named_parameters():
                save_weights[name] = (param.data.clone().detach().requires_grad_(True), param.grad)
            final_embed = embeddings.clone().detach().requires_grad_(True)
    
    model = Net(dim=Mdim).to(device)
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(save_weights[name][0])
            param.grad = save_weights[name][1]

    dic = dict()
    
    dic['Macc'] = Macc
    dic['Mdim'] = Mdim
    dic['epoch_vs_loss'] = epoch_vs_loss
    with open('NetB_train.pkl', 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    plt.plot(epoch_vs_loss)
    plt.savefig('epoch_vs_loss.png')
    
    torch.save(model.state_dict(), 'final_LSTM.pt')
    torch.save(final_embed, 'final_tuned_embed.pt')
