import torch
import pandas as pd
from main import CustomDataSet
from model import Net
from torch.utils.data import DataLoader

edim = 300 
dim = 150

model = Net(dim=dim, embed_dim=edim)
fpath = "Models/Embed-{}_Hidden-{}/model.pt".format(edim,dim)
model.load_state_dict(torch.load(fpath, map_location=torch.device('cpu')))
spath = "Models/Embed-{}_Hidden-{}/model_cpu.pt".format(edim,dim)
torch.save(model.state_dict(), spath)
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

vector = [float(0.1) for i in range(0,edim)] #Add another vector for unknown token(catchall for all Unknowns--UNK)
vec.append(vector)
embeddings = torch.tensor(vec, dtype=torch.float)

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

testdata = CustomDataSet(pd.read_csv('Test Dataset.csv'), embed=embedmap)
testloader = DataLoader(testdata, batch_size=100, num_workers=8, shuffle=False, collate_fn=collate_fn_padd)

acc = 0.0
for batch_idx, (Xb,lab,lend,lens) in enumerate(testloader):
    pre = model.predict(X=Xb, lend=lend.to(dtype=torch.int), lens=lens.to(dtype=torch.int)) #(N, )
    bacc = (torch.sum(pre==lab, dim=0))/(lab.shape[0])
    acc = acc + (bacc-acc)/(batch_idx+1)

print(acc)
fname = "Models/Embed-{}_Hidden-{}/Acc.txt".format(edim,dim)
with open(fname, 'w') as fil:
    L1 = "Embed dimension:{}\n".format(edim)
    L2 = "Hidden dimension:{}\n".format(dim)
    L3 = "Test Accuracy:{}".format(acc)
    fil.writelines([L1,L2,L3])
