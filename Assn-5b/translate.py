import re
import pickle
from typing import List
import torch
import pandas as pd
from queue import PriorityQueue
from main import vocab_transform, text_transform, nlp_preprocess
from model import Seq2SeqTransformer
from beamdecode import beam_decode

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SRC_LANGUAGE = 'wp' #Word Problem
TGT_LANGUAGE = 'pe' #Prefix expression

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol, beam=None):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask) #(S, E)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask) #(T, 1, E)
        import pdb; pdb.set_trace()
        out = out.transpose(0, 1) #(1, T, E)
        prob = model.generator(out[:, -1]) #(1, outvocab)
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_tokens: List[str], decode_fn=greedy_decode):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_tokens).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = decode_fn(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX, beam=6).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")


def fn(a: str):
    try:
        b = int(a)
    except:
        b = float(a)
    return b

def prefixeval(expr: List[str]):
    oper = {'+','*','/','-'} #Set of operators
    ret = 129971.1271123456753219 #Dummy return in case of exception thrown, bcoz of invalid prefix expression
    stack = []
    for i in range(len(expr)-1, -1, -1):
        if expr[i] not in oper:
             stack.append(fn(expr[i]))
        else:
            try:
                comp = None
                v1 = stack.pop()
                v2 = stack.pop()
                if expr[i] == '+':
                    comp = v1+v2
                elif expr[i] == '-':
                    comp = v1-v2
                elif expr[i] == '*':
                    comp = v1*v2
                else:
                    comp = v1/v2
                stack.append(comp)
            except:
                return ret
    ret = stack[0]
    return ret

            

"""For test, Change parameters here"""
ModelId = 2 #ModelId for the model under consideration
testdata = pd.read_excel('testdummy.xlsx') #Specify the File for Test here

path = "Models/Model{}".format(ModelId)
fpath = path + '/specs.pkl'
with open(fpath, 'rb') as f:
    dic = pickle.load(f)

SRC_VOCAB_SIZE = dic['SRC_VOCAB_SIZE']
TGT_VOCAB_SIZE = dic['TGT_VOCAB_SIZE']
EMB_SIZE = dic['EMB_SIZE']
NHEAD = dic['NHEAD']
FFN_HID_DIM = dic['FFN_HID_DIM']
NUM_ENCODER_LAYERS = dic['NUM_ENCODER_LAYERS']
NUM_DECODER_LAYERS = dic['NUM_DECODER_LAYERS']

fpath = path + '/model.pt'
transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM).to(DEVICE)

transformer.load_state_dict(torch.load(fpath, map_location=DEVICE))
nlkp = {} #number0 -> 0, number1 -> 1 and so on
for i in range(0,11):
    nlkp["number{}".format(i)] = i

log = [["V S S Aniketh"]]
for idx in range(0,len(testdata)):
    inp = nlp_preprocess(testdata.iloc[idx][0] +' . '+ testdata.iloc[idx][1]) #Assuming Description is in column0, and question in column1
    inp = inp.split()
    lkp = {}
    invlkp = {} #Inverse Look up
    ind = 0
    pat = "number[0-9]+"
    for k in inp:
        if re.match(pat,k):
            if k not in lkp:
                lkp[k] = ind
                invlkp["number{}".format(ind)] = k
                ind += 1
    
    for i in range(len(inp)):
        if re.match(pat, inp[i]):
            inp[i] = "number{}".format(lkp[inp[i]])

    outshuff = translate(model=transformer, src_tokens=inp, decode_fn=beam_decode) #Output for the evaluated expression in prefix format(shuffled version)
    
    """Get the desired Prefix expression after shuffling back the placeholders"""
    out = outshuff.split()
    for i in range(len(out)):
        if re.match(pat, out[i]):
            try:
                out[i] = invlkp[out[i]]
            except:
                out[i] = invlkp['number0']

    outprefix = " ".join(out)
    """outprefix, out are now actual prefix expr and their split respectively"""

    inum = testdata.iloc[idx][2].split() #List of strings(numbers in string format)

    oper = {'+','*','/','-'} #Set of operators
    pexpr = [i if i in oper else inum[nlkp[i]] for i in out]
    val = prefixeval(pexpr)
    log.append([outprefix, val])

df = pd.DataFrame(log)
name = "/outputhighb{}.xlsx".format(ModelId)
fname = path+name
df.to_excel(fname, sheet_name='Sheet1')   