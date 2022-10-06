#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 17:48:49 2022

@author: aniketh317
"""
import nltk 
import re
import string
import torch
import torch.nn as nn
import en_core_web_sm
import pickle
import os
from torch.utils.data import DataLoader,Dataset

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from random import shuffle
from model import Net
from train import train
import fasttext
import fasttext.util
from collections import OrderedDict

#ft = fasttext.load_model('cc.en.300.bin')
#fasttext.util.reduce_model(ft, 100)

nlp = en_core_web_sm.load()
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
torch.backends.cudnn.benchmark = True

dim = 300 #Word embedding dimension

N = 4 #Number of processes(Number of GPU devices being used)

#For creating a python dictionary between words and their glove embeddings(torch tensor form) 
embed_dict = {}
Path = '/raid/home/anikethv/others/glove.6B/glove.6B.{}d.txt'.format(dim)
with open(Path,'r') as f:
  for line in f:
    values = line.split()
    word = values[0]
    vector = torch.tensor([float(i) for i in values[1:]], dtype=torch.float)
    if len(vector)==dim:
        embed_dict[word]=vector

  
#Returning a preprocessed text, with 
def nlp_preprocess(text): 
    clean = re.compile('<.*?>')
    ret = re.sub(clean,'',text) #Remove HTML tags
    ret = "".join([i for i in ret if i not in string.punctuation]) #Remove punctuations
    ret = ret.lower() #Lower the case of letters
    ret = ret.split()
    ret = [i for i in ret if i not in nltk.corpus.stopwords.words('english')] #Remove stopwords
    ret = " ".join(ret) 
    return ret


train_neg_list  = [] #List of sentences with negative classification
with open('Train.neg','rb') as file:
    lines = file.readlines()
    train_neg_list = [[str(i) for i in nlp(nlp_preprocess(i.decode('utf-8', errors='ignore').rstrip()))] for i in lines]

train_pos_list  = [] #List of sentences with positive classification
with open('Train.pos','rb') as file:
    lines = file.readlines()
    train_pos_list = [[str(i) for i in nlp(nlp_preprocess(i.decode('utf-8', errors='ignore').rstrip()))] for i in lines]

test = []
for grams in train_neg_list:
    num_tokens = 0
    s = torch.zeros(dim, dtype=torch.float)
    for word in grams:
        if word in embed_dict:
            s += embed_dict[word]
            num_tokens += 1
    if num_tokens != 0:
        s = s/num_tokens
    test.append((s,torch.tensor(0)))

for grams in train_pos_list:
    num_tokens = 0
    s = torch.zeros(dim, dtype=torch.float)
    for word in grams:
        if word in embed_dict:
            s += embed_dict[word]
            num_tokens += 1
    if num_tokens != 0:
        s = s/num_tokens
    test.append((s,torch.tensor(1)))
    

L_proc = int(len(test)/N) #Length of data per process
test_split = [[test[i] for i in range(j*L_proc, (j+1)*L_proc)] for j in range(0,N)] #Split the data among N processes    

#Define Custom Dataset Loader class
class CustomDataset(Dataset):
    def __init__(self, data):
        self.X = data
    
    def __getitem__(self, idx):
        return self.X[idx]
    
    def __len__(self):
        return len(self.X)

def init_process(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)



wd = '0'
prob = '0.3'
lr = '0.01'

Path = '/raid/home/anikethv/others/DL-NLP/Prob-1/D2Prog-1/Models/model_{}_{}_{}'.format(wd, prob, lr)
fpath = Path + '/parameters.pt'

def worker(rank):
    init_process(rank, N, backend='nccl')
    model = Net(dim, float(prob))
    if rank == 0:
        par_dict = torch.load(fpath)
        new_par_dict = OrderedDict()
        for k, v in par_dict.items():
            name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
            new_par_dict[name] = v
        model.load_state_dict(new_par_dict)
    
    model = DDP(model.to(rank), device_ids=[rank]) # All parameters of model broadcasted from rank-0 process
    device = torch.device('cuda:{}'.format(rank))

    data = CustomDataset(test_split[rank])
    data_loader = DataLoader(data,batch_size=len(data),num_workers=0, shuffle=False) #batch-loader for testing

    model.eval()
    data_iter = iter(data_loader)
    (test_X, test_y) = next(data_iter)

    test_pred = model.module.predict(test_X.to(device))
    acc = sum(test_pred == test_y.to(device))/len(data)

    dist.all_reduce(acc)
    acc = acc/N
    print(acc)

if __name__ == "__main__":
    mp.spawn(fn=worker, nprocs=N)

    
    

    
    

