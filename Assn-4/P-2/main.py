import string
import nltk 
import re
import numpy as np
import torch
import torch.nn as nn
import pickle
import pandas as pd
from torch.utils.data import Dataset
from train import model_train
from model import Net
import os

from  matplotlib import pyplot as plt

def nlp_preprocess(text): 
    global max_len
    clean = re.compile('<.*?>')
    ret = re.sub(clean,'',text) #Remove HTML tags
    ret = "".join([i if i not in string.punctuation else " " for i in ret]) #Remove punctuations
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
        X = self.data.iloc[idx]['review']
        Xs = X.split(".") #Split the document into list of sentences, about "."
        sl = []
        for s in Xs:
            ret = nlp_preprocess(s)
            if len(ret)!=0:
                sl.append(ret)
        
        X = [[self.embed[z] if z in self.embed else -1 for z in s.split()] for s in sl] # List of sentences in doc, The last embedding corresponds to UNK
        
        if self.data.iloc[idx]['sentiment'] == 'positive':
            lab = 1
        else:
            lab = 0
        
        ld = len(X) #Length of document
        ls = [len(s) for s in X]
        sample = (X,lab,ld,ls)
        return sample



embeddings = None
edim_list = [100,200,300]
dim_list = [50,150,300]

ed_comb = [(e,d) for e in edim_list for d in dim_list]

#batch(input to this function) would be a list of tuples corresponding to the indices used by dataloader
def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    B = len(batch)
    Ms = max([t[2] for t in batch])
    Mw = max([max(t[3]) for t in batch])
    Xb = torch.zeros(B,Ms,Mw,edim) #All Zeros are used for paddings
    lab = torch.tensor([t[1] for t in batch])
    lend = torch.zeros(B, dtype=torch.int)
    lens = torch.ones(B,Ms, dtype=torch.int) #lens[b][m] is the length of mth sentence in bth doc in batch
    for b_idx,t in enumerate(batch):
        (X,label,ld,ls) = t #ls is list of sentence lengths in this doc
        lend[b_idx] = ld #The document size for this doc in the batch(no. of sentences in the doc)
        for s in range(ld):
            lens[b_idx,s] = ls[s] #Length of this sentence
            emb = embeddings[X[s]]
            Xb[b_idx,s,0:ls[s],:] = emb

    return (Xb,lab,lend,lens)


criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':
    torch.cuda.empty_cache()
    rank = 8 #Change this rank for different train tasks (0-8) is possible range
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:1" if use_cuda else "cpu") #if gpu present cuda:0 else cpu only
    torch.backends.cudnn.benchmark = True
    Macc = None
    Mdim = None
    model = None
    save_weights = {}
    final_embed = None
    
    (edim,dim) = ed_comb[rank]
    #For creating a python dictionary between words and their glove embeddings(torch tensor form) 
    Path = '/raid/home/anikethv/glove/glove.6B/glove.6B.{}d.txt'.format(edim)
    i=0
    vec = []
    embedmap = dict()

    with open(Path,'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = [float(i) for i in values[1:]]
            if len(vector)!=edim:
                vector = [float(0.1) for i in range(0,edim)]
            vec.append(vector)
            embedmap[word] = i
            i+=1
    
    path = "Models/Embed-{}_Hidden-{}".format(edim,dim)
    if not os.path.exists(path):
        os.makedirs(path)

    vector = [float(0.1) for i in range(0,edim)] #Add another vector for unknown token(catchall for all Unknowns--UNK)
    vec.append(vector)
    embeddings = torch.tensor(vec, dtype=torch.float)
    tdata = CustomDataSet(pd.read_csv('Train dataset.csv'), embed=embedmap)
    print("TRAIN START FOR RANK = {}".format(rank))
    print('Training start for Edim, Dim : {}'.format(edim,dim))
    model = Net(dim=dim, embed_dim=edim).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.001)
    vacc, epoch_vs_loss = model_train(model, train_data=tdata, criterion=criterion, optimiser=optimiser, collate_fn=collate_fn_padd, 
                                      embed=edim, hidden=dim, device=device, verbose=True)
    print('\n\nAccuracy on Validation set:{}\n'.format(vacc))
    p = path + '/model.pt'
    torch.save(model.state_dict(), p)

    dic = dict()
    dic['vacc'] = vacc
    dic['epoch_vs_loss'] = epoch_vs_loss
    p = path + '/metrics.pkl'
    with open(p, 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.plot(epoch_vs_loss)
    plt.savefig(path+'epoch_vs_loss.png')
