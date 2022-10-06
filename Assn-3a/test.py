import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from model import Net
from main import nlp_preprocess
#For creating a python dictionary between words and their glove embeddings(torch tensor form)
dim = 200
Path = '/raid/home/anikethv/others/glove.6B/glove.6B.{}d.txt'.format(dim)
i=0
vec = []
embedmap = dict()


#Load the model, and embeddings
model = Net(dim=dim)
model.load_state_dict(torch.load('final_GRU.pt', map_location=torch.device('cpu')))

torch.save(model.state_dict(), 'final_GRU_cpu.pt')
#torch.save(embeddings, 'final_tuned_embed_cpu.pt')

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


batch = 400
testdata = CustomDataSet(pd.read_csv('Test Dataset.csv'), embed=embedmap)
testloader = DataLoader(testdata, shuffle=False, batch_size=batch, collate_fn=collate_fn_padd)

acc = 0.0
for batch_idx, (Xb, lab) in enumerate(testloader):
    pred = model.predict(Xb)
    bacc = (torch.sum(pred==lab))/lab.shape[0]
    acc = acc + (bacc*(lab.shape[0]/batch) - acc)/(batch_idx+1)

print(acc)