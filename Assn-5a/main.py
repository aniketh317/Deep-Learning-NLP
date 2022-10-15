import string
import nltk 
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
from train import model_train
from model import Net
import os

from  matplotlib import pyplot as plt

def nlp_preprocess(text): 
    global max_len
    clean = re.compile('<.*?>')
    ret = re.sub(clean,' ',text) #Remove HTML tags
    ret = "".join([i if i not in string.punctuation else " " for i in ret]) #Remove punctuations
    ret = ret.lower() #Lower the case of letters
    ret = ret.split()
    ret = [i for i in ret if i not in nltk.corpus.stopwords.words('english') and i!='not'] #Remove stopwords but keep not
    ret = " ".join(ret) 
    return ret

class CustomDataSet(Dataset):
    def __init__(self, data, edim=100):
        """
        data --> Test/Train data in Sentence-Review form
        edim --> Embed dimension
        """
        self.data = data

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
                    vector = [float(0.1) for _ in range(0,edim)]
                vec.append(vector)
                embedmap[word] = i
                i+=1
        vector = [float(0.1) for _ in range(0,edim)] #Add another vector for unknown token(catchall for all Unknowns--UNK, PADS)
        vec.append(vector)

        self.embedmap = embedmap
        self.embeddings = torch.tensor(vec, dtype=torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        #use word embedding(glove) to encode sentences 
        X = nlp_preprocess(self.data.iloc[idx]['review'])
        X = X.split()
        vocab = self.embeddings.shape[0]
        X = [self.embedmap[z] if z in self.embedmap else vocab-1 for z in X] #vocab-1 corresponds to UNK token and PAD token
        
        if self.data.iloc[idx]['sentiment'] == 'positive':
            lab = 1
        else:
            lab = 0

        sample = (X,lab,vocab)
        return sample

#batch(input to this function) would be a list of tuples corresponding to the indices used by dataloader
def collate_fn_padd(batch):
    '''
    Pads batch of variable length and returns a PAD/UNK mask
    '''
    vocab = batch[0][2]
    seqs = [torch.tensor(t[0]) for t in batch]
    seqs = pad_sequence(seqs, batch_first=True, padding_value=vocab-1) #Pad with (vocab-1), the index for the PAD and UNK tokens in the embeddings (N, max_len)
    mask = torch.where(seqs == (vocab-1), True, False) #mask the UNK tokens and PAD tokens (N, max_len)
    labs = torch.tensor([t[1] for t in batch])
    return seqs, labs, mask

criterion = nn.CrossEntropyLoss()
embed_heads = [(100,5), (200,5), (200,8), (300,5), (300,6)]

if __name__ == '__main__':
    rank = 0 #Change this rank for different train tasks (0-8) is possible range
    
    # CUDA for PyTorch
    gpu_count = torch.cuda.device_count()
    device = torch.device('cpu')
    if gpu_count>0:
        device = torch.device('cuda:{}'.format(rank%gpu_count))
    torch.backends.cudnn.benchmark = True

    Macc = None
    Mdim = None
    model = None
    save_weights = {}
    
    (edim,num_heads) = embed_heads[rank]
    
    path = "Models/Embed-{}_Heads-{}".format(edim,num_heads)
    if not os.path.exists(path):
        os.makedirs(path)

    tdata = CustomDataSet(pd.read_csv('Train dataset.csv'), edim=edim)
    print("TRAIN START FOR RANK = {}".format(rank))
    print('Training start for Embed:{}, Num-Heads:{}'.format(edim,num_heads))
    model = Net(init_embedding=tdata.embeddings, n_heads=num_heads).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)
    vacc, epoch_vs_loss = model_train(model, train_data=tdata, criterion=criterion, optimiser=optimiser, collate_fn=collate_fn_padd, 
                                      embed=edim, heads=num_heads, device=device, verbose=True)
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