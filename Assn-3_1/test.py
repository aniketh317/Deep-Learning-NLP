import nltk 
import re
import string
import torch
import en_core_web_sm
from torch.utils.data import DataLoader,Dataset
import numpy as np

from model import Net


nlp = en_core_web_sm.load()
dim = 300
# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu") #if gpu presesnt cuda:0 else cpu only

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

test = []
k = 0
with open('TestData', 'rb') as file:
    lines = file.readlines()
    for l in lines:
        if k<331:
            test.append((" ".join([str(i) for i in nlp(nlp_preprocess(l.decode('utf-8', errors='ignore').rstrip()))]),1)) #For positive examples
        else:
            test.append((" ".join([str(i) for i in nlp(nlp_preprocess(l.decode('utf-8', errors='ignore').rstrip()))]),0)) #For negative examples
        k += 1

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
        try:
            tor = torch.tensor(t[0])
        except:
            print(t[0])
            raise Exception

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

testdata = CustomDataSet(data=test, embed=embed_dict)

batch = 662
testloader = DataLoader(testdata, batch_size=662, num_workers=0, shuffle=False, collate_fn=collate_fn_padd)

model = Net(dim).to(device)
model.load_state_dict(torch.load('final_GRU.pt'))

acc = 0.0
for batch_idx, (Xb, lab) in enumerate(testloader):
    Xb = Xb.to(device)
    lab = lab.to(device)
    pred = model.predict(Xb)
    bacc = (torch.sum(pred==lab))/lab.shape[0]
    acc = acc + (bacc - acc)/(batch_idx+1)

print(acc)