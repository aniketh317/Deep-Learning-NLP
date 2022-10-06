import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.multiprocessing as mp

from model import Net
from train import model_train
from vocab import human_vocab, machine_vocab, inv_machine_vocab


torch.backends.cudnn.benchmark = True

train_data  = [] #List of sentences with negative classification
with open('Assignment4aDataset.txt', 'rb') as file:
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
        train_data.append(z)

class CustomData(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)

def collate_fn_train(batch):
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

edim = [10,15,20,30]
dim = [30,50,75,100]
edimdim = [(e,d) for e in edim for d in dim]


def worker(rank, offset, epoch_s):
    """Worker for training the models"""
    """epoch_s --> start epoch"""
    e,d = edimdim[rank+offset]

    gpu_count = torch.cuda.device_count()
    device = torch.device('cpu')
    if gpu_count>0:
        device = torch.device('cuda:{}'.format(rank%gpu_count))
    
    model = Net(dim=d, num_outclass=len(machine_vocab), num_inclass=len(human_vocab), embed_dim=e).to(device)
    
    path = "Models/Embed-{}_Hidden-{}".format(e,d)
    epoch_vs_loss_old = []
    if epoch_s>0:
        fname = path+"/"+"model.pt"
        model.load_state_dict(torch.load(fname))
        fname = path+"/"+"metrics.pkl"
        with open(fname, 'rb') as handle:
            sav = pickle.load(handle)
            epoch_vs_loss_old = sav['epoch_vs_loss']
    
    tdata = CustomData(train_data)
    loss_function = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adagrad(model.parameters(), lr=0.001) 
    valid_acc, valid_acc_time, epoch_vs_loss = model_train(model=model, train_data=tdata, criterion=loss_function, optimiser=optimiser, collate_fn=collate_fn_train,
                                            device=device, embed=e, hidden=d, epoch_l=800, epoch_s=0, verbose=True)

    epoch_vs_loss = epoch_vs_loss_old + epoch_vs_loss
    if not os.path.exists(path):
        os.makedirs(path)
    
    sav = {'valid_acc':valid_acc, 'valid_acc_time':valid_acc_time, 'epoch_vs_loss':epoch_vs_loss}
    fname = path+"/"+"metrics.pkl"
    with open(fname, 'wb') as handle:
        pickle.dump(sav, handle, protocol=pickle.HIGHEST_PROTOCOL)
    fname = path+"/"+"model.pt"
    torch.save(model.state_dict(), fname)

if __name__ == "__main__":
    worker(3,0,0)
