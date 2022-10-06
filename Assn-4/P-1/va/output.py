import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import Net
from train import model_train
from vocab import human_vocab, machine_vocab, inv_machine_vocab

test_data  = [] #List of sentences with negative classification
with open('Assignment4aTestDataset.txt', 'rb') as file:
    lines = file.readlines()
    for l in lines:
        z = l.decode('utf-8', errors='ignore')
        z = z.rstrip() #Remove newlines and spaces at the ends
        z = z.replace("'","")
        z = z.split(",") #Split around ","
        z[0] = z[0].lstrip()
        z[0] = z[0].rstrip()
        z[1] = z[1].lstrip()
        z[1] = z[1].rstrip()
        z1 = [human_vocab[k] if k in human_vocab else human_vocab['<unk>'] for k in [*z[0]]] #Split each sentence into characters [source, target]
        z2 = [machine_vocab[k] if k in machine_vocab else machine_vocab['END'] for k in [*z[1]]]
        z = [z1,z2]
        test_data.append(z)

class CustomData(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)

def collate_fn_test(batch):
    lengths = torch.tensor([len(t[0]) for t in batch])
    lmax = torch.max(lengths)
    X = None
    Y = None
    for t in batch:
        Xt = (torch.tensor([t[0][i] if i<len(t[0]) else human_vocab['<pad>'] for i in range(0, lmax)])).unsqueeze(dim=0)
        Yt = (torch.tensor([t[1][i] if i<10 else machine_vocab['END'] for i in range(0, 11)])).unsqueeze(dim=0)

        if X is not None:
            X = torch.cat((X,Xt), dim=0)
        else:
            X = Xt
        
        if Y is not None:
            Y = torch.cat((Y,Yt), dim=0)
        else:
            Y = Yt
    return (X, Y, lengths) #X->(N, lmax), Y->(N, lmax), length->(N,)

inv_human_vocab = {}
for k,v in human_vocab.items():
    inv_human_vocab[v] = k

def logger(file, X, y):
    for i in range(X.shape[0]):
        stext = "".join([inv_human_vocab[k.item()] for k in X[i] if k.item()!=35 and k.item()!=36])
        outext = "".join([inv_machine_vocab[k.item()] for k in y[i] if k.item()!=11])
        file.write(stext+":::"+outext+"\n")
    return
        
dim = 100
embed_dim = 10
model = Net(dim=dim, num_outclass=len(machine_vocab), num_inclass=len(human_vocab), embed_dim=embed_dim)
path = 'Models/Embed-{}_Hidden-{}'.format(embed_dim, dim)

model.load_state_dict(torch.load(path+'/model.pt', map_location=torch.device('cpu')))

testdata = CustomData(test_data)
test_loader = DataLoader(testdata, batch_size=500, shuffle=False, collate_fn=collate_fn_test)

tacc = 0.0
tacc_time = torch.tensor([0.0 for i in range(0,11)])
tloader = DataLoader(test_data, batch_size=250, num_workers=8, shuffle=False, collate_fn=collate_fn_test)
match = 0.0
for batch_idx, (Xb,lab,lens) in enumerate(tloader):
    pre = model.predict(Xb, lens) #(N,11)
    err = torch.sum(pre!=lab, dim=1)
    match_b = (lab.shape[0]-(torch.nonzero(err).shape[0]))/lab.shape[0]
    bacc_time = (torch.sum(pre==lab, dim=0))/(lab.shape[0])
    tacc_time = tacc_time + (bacc_time-tacc_time)/(batch_idx+1)
    bacc = torch.sum(bacc_time[0:10])/10
    tacc = tacc + (bacc-tacc)/(batch_idx+1)
    match = match + (match_b-match)/(batch_idx+1)

print("Test Accuracy : {}".format(tacc))
print("Test Accurancy for each time stamp:{}".format(tacc_time))
print('Excat Match:{}'.format(match))