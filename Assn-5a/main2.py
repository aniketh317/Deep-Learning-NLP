import string
import nltk 
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pickle
import pandas as pd
from train1 import model_train
from model2 import Net
import os

from  matplotlib import pyplot as plt

from transformers import BertTokenizer

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
    def __init__(self, data):
        """
        data --> Test/Train data in Sentence-Review form
        edim --> Embed dimension
        """
        self.data = data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        #use word embedding(glove) to encode sentences 
        X = nlp_preprocess(self.data.iloc[idx]['review'])

        if self.data.iloc[idx]['sentiment'] == 'positive':
            lab = 1
        else:
            lab = 0

        sample = (X,lab)
        return sample


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

#batch(input to this function) would be a list of tuples corresponding to the indices used by dataloader
def collate_fn_padd(batch):
    '''
    Pads batch of variable length and returns a PAD/UNK mask
    '''
    sens = [t[0] for t in batch]
    maxlen = max([len(t[0].split()) for t in batch])
    if maxlen>508:
        maxlen = 508
    
    labs = torch.tensor([t[1] for t in batch])
    # Encode the sentence

    attention_masks = []
    input_ids = []
    for text in sens:
        encoded = tokenizer.encode_plus(
        text=text,  # the sentence to be encoded
        add_special_tokens=True,  # Add [CLS] and [SEP]
        max_length = maxlen,  # maximum length of a sentence
        pad_to_max_length=True,  # Add [PAD]s
        return_attention_mask = True,  # Generate the attention mask
        return_tensors = 'pt',  # ask the function to return PyTorch tensors
        truncation=True
        )
        attention_masks.append(encoded['attention_mask'])
        input_ids.append(encoded['input_ids'])
    
    input_ids = torch.cat(input_ids)
    attention_masks = torch.cat(attention_masks) 

    return input_ids, labs, attention_masks 

criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':
    
    # CUDA for PyTorch
    gpu_count = torch.cuda.device_count()
    device = torch.device('cpu')
    if gpu_count>0:
        device = torch.device('cuda:{}'.format(3))
    torch.backends.cudnn.benchmark = True

    model = None
    save_weights = {}
    
    
    path = "Models/BERT_NO_TUNE"
    if not os.path.exists(path):
        os.makedirs(path)

    tdata = CustomDataSet(pd.read_csv('Train dataset.csv'))
    print("TRAIN START FOR BERT_FINE_TUNE")
    model = Net().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=0.0001)
    vacc, epoch_vs_loss = model_train(model, train_data=tdata, criterion=criterion, optimiser=optimiser, collate_fn=collate_fn_padd, 
                                      device=device, verbose=True)
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