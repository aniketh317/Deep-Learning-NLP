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
from random import shuffle
from model import Net
from train import train


nlp = en_core_web_sm.load()
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu") #use cuda:0
torch.backends.cudnn.benchmark = True

dim = 50 #Word embedding dimension

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

data_train = []
for grams in train_neg_list:
    num_tokens = 0
    s = torch.zeros(dim, dtype=torch.float)
    for word in grams:
        if word in embed_dict:
            s += embed_dict[word]
            num_tokens += 1
    if num_tokens != 0:
        s = s/num_tokens
    data_train.append((s,torch.tensor(0)))

for grams in train_pos_list:
    num_tokens = 0
    s = torch.zeros(dim, dtype=torch.float)
    for word in grams:
        if word in embed_dict:
            s += embed_dict[word]
            num_tokens += 1
    if num_tokens != 0:
        s = s/num_tokens
    data_train.append((s,torch.tensor(1)))
    
shuffle(data_train) #shuffle the training list as positive and negative samples are arranged together    

#Define Custom Dataset Loader class
class CustomDataset(Dataset):
    def __init__(self, train_data):
        self.X = train_data
    
    def __getitem__(self, idx):
        return self.X[idx]
    
    def __len__(self):
        return len(self.X)

#Create a datset out of Train_data
Data = CustomDataset(data_train)

#75-25 split of training data ->(train,validate)
train_size = int(0.75*len(Data))
valid_size = len(Data)-train_size
train_data,valid_data = torch.utils.data.random_split(Data,[train_size,valid_size]) #dataset subset objects created

#Create Loaders/Data generators
"""Default collate_fn collates the received tuples tuples received st. the first components are
 collated together and seconds components are collated together to form a final tuple which is returned"""
 
train_loader = DataLoader(train_data, batch_size=16, num_workers=0, shuffle=True) #batch-loader for traindata
valid_loader = DataLoader(valid_data,batch_size=len(valid_data),num_workers=0, shuffle=False) #batch-loader for validation

#Define Loss criterion(Cross-Entropy Loss)
criterion = nn.CrossEntropyLoss()

prob_list = [0,0.1,0.2,0.3] #list of probabilty dropouts for dropout layer
lr_list = [0.001, 0.01, 0.1] #List of learning rates being chosen(choose best for validation)


Path = '/raid/home/anikethv/others/DL-NLP/Prog-1/Models/'

min_valid_loss = 1000.000
lr_sel = -1
prob_sel = -1

for prob in prob_list:
    for lr in lr_list:
        print('Starting training and evaluation for prob={}, lr={}'.format(prob,lr))
        model = Net(dim, prob).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        epochs, valid_loss = train(model=model, train_loader=train_loader, validation_loader=valid_loader, loss_criterion=criterion, optimizer=optimizer, verbose=True)
        directory = Path+'model_{}_{}/'.format(prob, lr)
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = directory + 'parameters.pt'
        torch.save(model.state_dict(), file_path)

        file_path = directory + 'EL.pickle'
        a = {'epochs':epochs, 'valid_loss':valid_loss}
        with open(file_path,'wb') as handle:
            pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        if valid_loss < min_valid_loss:
            lr_sel = lr
            prob_sel = prob

model = Net(dim, prob_sel).to(device)
file_path = Path + 'model_{}_{}/'.format(prob_sel, lr_sel) + 'parameters.pt' 
model.load_state_dict(torch.load(file_path))

model.eval()

valid_iter = iter(valid_loader)
(valid_X, valid_y) = next(valid_iter)
valid_pred = model.predict(valid_X.to(device))
acc = sum(valid_pred == valid_y.to(device))/len(valid_data)
print(acc)
file_path = Path + 'Hyper_Acc.pickle'
with open(file_path,'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
