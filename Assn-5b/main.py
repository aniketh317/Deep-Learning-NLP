import os
import string
import re
from typing import Iterable, List
import pandas as pd
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from model import Seq2SeqTransformer
from train import train_epoch, evaluate

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SRC_LANGUAGE = 'wp' #Word Problem
TGT_LANGUAGE = 'pe' #Prefix expression 

# Place-holders
vocab_transform = {}

def nlp_preprocess(text): 
    global max_len
    clean = re.compile('<.*?>')
    ret = re.sub(clean,' ',text) #Remove HTML tags
    ret = "".join([i if i not in string.punctuation else " " for i in ret]) #Remove punctuations
    ret = ret.lower() #Lower the case of letters 
    return ret

tgt_vocab = ['+','-','*','/',
'number0',
'number1',
'number2',
'number3',
'number4',
'number5',
'number6',
'number7',
'number8',
'number9',
'number10']

data = pd.read_excel('ArithOpsTrain.xlsx')
inseq = []
outseq = []

for idx in range(1,len(data)):
    inp = nlp_preprocess(data.iloc[idx][1] +' . '+ data.iloc[idx][2])
    inp = inp.split()
    lkp = {}
    ind = 0
    pat = "number[0-9]+"
    for k in inp:
        if re.match(pat,k):
            if k not in lkp:
                lkp[k] = ind
                ind += 1
    
    for i in range(len(inp)):
        if re.match(pat, inp[i]):
            inp[i] = "number{}".format(lkp[inp[i]])
    inseq.append(inp)
    try:
        out = data.iloc[idx][3]
        out = out.split()
        for i in range(len(out)):
            if re.match(pat, out[i]):
                out[i] = "number{}".format(lkp[out[i]])
        outseq.append(out)
    except:
        print(idx)
        raise Exception

train = list(zip(inseq, outseq))


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


inseq = inseq+['number0',
'number1',
'number2',
'number3',
'number4',
'number5',
'number6',
'number7',
'number8',
'number9',
'number10'] #To make sure that these tokens are there entirely in vocab for the input


# helper function to yield list of tokens
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter[language_index[language]]:
        yield data_sample

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data Iterator
    train_iter = [inseq, outseq]
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

# Set UNK_IDX as the default index. This index is returned when the token is not found.
# If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  vocab_transform[ln].set_default_index(UNK_IDX)

from torch.nn.utils.rnn import pad_sequence

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# src and tgt language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor


# function to collate data samples into batch tesors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


torch.manual_seed(0)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 32
NHEAD = 4
FFN_HID_DIM = 128
BATCH_SIZE = 64
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3


loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

from timeit import default_timer as timer

NUM_EPOCHS = 100 #Maximum number of epochs

if __name__ == '__main__':
    i = -1
    clash = True
    path = None
    while clash:
        i += 1
        path = "Models/Model{}".format(i)
        clash = os.path.exists(path)
    
    os.makedirs(path)

    dic = dict()

    fpath = path + '/specs.pkl'
    dic['SRC_VOCAB_SIZE'] = SRC_VOCAB_SIZE
    dic['TGT_VOCAB_SIZE'] = TGT_VOCAB_SIZE
    dic['EMB_SIZE'] = EMB_SIZE
    dic['NHEAD'] = NHEAD
    dic['FFN_HID_DIM'] = FFN_HID_DIM
    dic['BATCH_SIZE'] = BATCH_SIZE
    dic['NUM_ENCODER_LAYERS'] = NUM_ENCODER_LAYERS
    dic['NUM_DECODER_LAYERS'] = NUM_DECODER_LAYERS
    dic['SRC_VOCAB'] = vocab_transform[SRC_LANGUAGE]
    dic['TGT_VOCAB'] = vocab_transform[TGT_LANGUAGE]

    with open(fpath, 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    data = CustomDataset(train)
    
    #90-10 split train,valid
    tlen = int(0.90*len(data))
    vlen = len(data)-tlen
    lengths = [tlen, vlen]
    tdata, vdata = torch.utils.data.random_split(data, lengths)
    val_loss = evaluate(transformer, loss_fn=loss_fn, collate_fn=collate_fn, valid_data=vdata)
    mloss = val_loss

    epoch = 0
    stop = False
    patience = 25
    obs = 0
    save_weights = {}

    print((f"Epoch: {epoch}, Val loss: {val_loss:.3f}"))
    while epoch <= NUM_EPOCHS and not stop:
        epoch += 1
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, loss_fn=loss_fn, collate_fn=collate_fn, train_data=tdata)
        end_time = timer()
        val_loss = evaluate(transformer, loss_fn=loss_fn, collate_fn=collate_fn, valid_data=vdata)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
        if val_loss < mloss:
            mloss = val_loss
            for name, param in transformer.named_parameters():
                save_weights[name] = (param.data.clone().detach().requires_grad_(True), param.grad)
            num_epoch = epoch
            obs = 0
        else:
            obs += 1
            if obs > patience:
                stop = True
    
    with torch.no_grad():
        for name, param in transformer.named_parameters():
            param.copy_(save_weights[name][0])
            param.grad = save_weights[name][1]
    

    fpath = path + '/model.pt'
    torch.save(transformer.state_dict(), fpath)

    
