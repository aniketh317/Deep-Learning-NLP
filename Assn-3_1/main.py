import string
import nltk 
import re
import numpy as np
import torch
import torch.nn as nn
import pickle
import pandas as pd
import en_core_web_sm

from torch.utils.data import Dataset
from train import model_train
from model import Net

from random import shuffle

from  matplotlib import pyplot as plt

nlp = en_core_web_sm.load()
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu") #if gpu presesnt cuda:0 else cpu only
torch.backends.cudnn.benchmark = True


def nlp_preprocess(text):
    clean = re.compile('<.*?>')
    ret = re.sub(clean,'',text) #Remove HTML tags
    ret = "".join([i for i in ret if i not in string.punctuation]) #Remove punctuations
    ret = ret.lower() #Lower the case of letters
    ret = ret.split()
    ret = [i for i in ret if i not in nltk.corpus.stopwords.words('english') and i!='not'] #Remove stopwords but keep not
    ret = " ".join(ret) 
    return ret

train_neg_list  = [] #List of sentences with negative classification
with open('Train.neg','rb') as file:
    lines = file.readlines()
    train_neg_list = [(" ".join([str(i) for i in nlp(nlp_preprocess(i.decode('utf-8', errors='ignore').rstrip()))]),0) for i in lines]

train_pos_list  = [] #List of sentences with positive classification
with open('Train.pos','rb') as file:
    lines = file.readlines()
    train_pos_list = [(" ".join([str(i) for i in nlp(nlp_preprocess(i.decode('utf-8', errors='ignore').rstrip()))]),1) for i in lines]

data_train = train_neg_list + train_pos_list
shuffle(data_train) #shuffle the training list as positive and negative samples are arranged together

class CustomDataSet(Dataset):
    def __init__(self, data, embed):
        self.data = data
        self.embed = embed
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        
        #use word embedding(glove) to encode sentences 
        X = nlp_preprocess(self.data[idx][0])
        X = X.split()
        
        X = [self.embed[z] if z in self.embed and len(self.embed[z]) == dim else list(np.ones(dim)*0.1) for z in X]
        
        lab = self.data[idx][1]
        sample = (X,lab)
        return sample





#batch(input to this function) would be a list of tuples corresponding to the indices used by dataloader
def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    
    #pack the padded
    X_l = []
    X_len = []
    for t in batch:
        tor = torch.tensor(t[0])
        if tor.ndim ==2 and tor.shape[1] == dim:
            X_l.append(tor.float())
            X_len.append(tor.shape[0])
        else:
            print('stale index',t[2])
            X_l.append((torch.ones(1,dim)*0.1).float())
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
    for dim in [50, 100, 200, 300]:
        #For creating a python dictionary between words and their glove embeddings(torch tensor form) 
        embed_dict = {}
        Path = '/raid/home/anikethv/others/glove.6B/glove.6B.{}d.txt'.format(dim)
        with open(Path,'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = [float(i) for i in values[1:]]
                if len(vector)!=dim:
                    vector = [float(0.1) for i in range(0,dim)]
                embed_dict[word]=vector
        
        tdata = CustomDataSet(data=data_train, embed=embed_dict)
        print('Training start for Dimension : {}'.format(dim))
        model = Net(dim=dim).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
        vacc, epoch_vs_loss = model_train(model, train_data=tdata, criterion=criterion, optimiser=optimiser, verbose=True, 
                                    collate_fn=collate_fn_padd)
        print('\n\nAccuracy on Validation set:{}\n'.format(vacc))
        p = 'GRU_{}.pt'.format(dim)
        torch.save(model.state_dict(), p)
        if Macc is None or vacc > Macc:
            Macc = vacc
            Mdim = dim
            for name, param in model.named_parameters():
                save_weights[name] = (param.data.clone().detach().requires_grad_(True), param.grad)
    
    model = Net(dim=Mdim).to(device)
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(save_weights[name][0])
            param.grad = save_weights[name][1]

    dic = dict()
    
    dic['Macc'] = Macc
    dic['Mdim'] = Mdim
    dic['epoch_vs_loss'] = epoch_vs_loss
    with open('Net_train_gru.pkl', 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    plt.plot(epoch_vs_loss)
    plt.savefig('epoch_vs_loss_gru.png')
    
    torch.save(model.state_dict(), 'final_GRU.pt')