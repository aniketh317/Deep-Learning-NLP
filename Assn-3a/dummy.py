import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from model import Net
from main import nlp_preprocess
#For creating a python dictionary between words and their glove embeddings(torch tensor form)
dim = 200
Path = '/raid/home/anikethv/others/glove.6B/glove.6B.{}d.txt'.format(dim)
i=0
vec = []
embedmap = dict()

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu") #if gpu presesnt cuda:0 else cpu only

#Load the model, and embeddings
model = Net(dim=dim).to(device)
model.load_state_dict(torch.load('final_GRU.pt'))


with open(Path,'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = [float(i) for i in values[1:]]
        if len(vector)!=dim:
            vector = [float(0.1) for i in range(0,dim)]
        vec.append(vector)
        embedmap[word] = vector

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
        
        X = [self.embed[z] if z in self.embed and len(self.embed[z]) == dim else list(np.ones(dim)*0.1) for z in X]
        
        if self.data.iloc[idx]['sentiment'] == 'positive':
            lab = 1
        else:
            lab = 0

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

data1 = CustomDataSet(pd.read_csv('Train dataset.csv'), embed=embedmap)
data2 = CustomDataSet(pd.read_csv('Test Dataset.csv'), embed=embedmap)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
epoch_l = 3
epoch = 0

while epoch<epoch_l:
    load1 = DataLoader(data1, collate_fn=collate_fn_padd, batch_size=256, num_workers=8,shuffle=True)
    load2 = DataLoader(data2, collate_fn=collate_fn_padd, batch_size=256, num_workers=8,shuffle=True)
    loss = 0.0
    for batch_idx, (Xb, lab) in enumerate(load1):
        optimizer.zero_grad()
        b_loss = criterion(model(Xb.to(device)), lab.to(device))
        b_loss.backward()
        optimizer.step()

    for batch_idx, (Xb, lab) in enumerate(load2):
        optimizer.zero_grad()
        b_loss = criterion(model(Xb.to(device)), lab.to(device))
        b_loss.backward()
        optimizer.step()
    
    epoch += 1

#Change Test Data here
batch = 400
testloader = DataLoader(data2, shuffle=False, batch_size=batch, collate_fn=collate_fn_padd, num_workers=6)

acc = 0.0
for batch_idx, (Xb, lab) in enumerate(testloader):
    Xb = Xb.to(device)
    lab = lab.to(device)
    pred = model.predict(Xb)
    bacc = (torch.sum(pred==lab))/lab.shape[0]
    acc = acc + (bacc*(lab.shape[0]/batch) - acc)/(batch_idx+1)

print(acc)

torch.save(model.state_dict(), 'final_GRU22.pt')    
