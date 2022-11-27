import sys
sys.path.append('../')

import pickle
import torch
from mono.data.pipe import ENBertPipe
from fastNLP import cache_results
from transformers import BertTokenizer, RobertaTokenizer
import pandas as pd
from torch.nn.utils.rnn import pad_sequence


max_word_len = 5
paths = '../data/en'

model_lkp = {}
model_lkp['bert'] = ['./save_models/best_ENBertReverseDict_t10_2022-11-26-14-04-43', BertTokenizer.from_pretrained('bert-base-uncased'),'bert-base-uncased']
model_lkp['roberta'] = ['.',RobertaTokenizer.from_pretrained('roberta-base'),'roberta-base']


model_sel = 'bert'


mpath = model_lkp[model_sel][0]
tokenizer = model_lkp[model_sel][1]
pre_name = model_lkp[model_sel][2]

@cache_results('./caches/en_{}_{}.pkl'.format(pre_name.split('/')[-1], max_word_len), _refresh=False)
def get_data():
    data_bundle = ENBertPipe(pre_name, max_word_len).process_from_file(paths)
    return data_bundle

data_bundle = get_data()
word2bpes = data_bundle.word2bpes
pad_id = data_bundle.pad_id

word2idx = data_bundle.word2idx
idx2word = list(range(len(word2idx)))
for w,v in word2idx.items():
    idx2word[v] = w


model = torch.load(mpath)

mask_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
sep_id = tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
cls_id = tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
pad_id = tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
unk_id = tokenizer.convert_tokens_to_ids(['[UNK]'])[0]

with open('lookup.pkl','rb') as file:
    lookup = pickle.load(file)

with open('testinputs.txt','r') as file:
    lines = file.readlines()

log = []
in_list = []
sen_list = []
for l in lines:
    sentence = l.strip()
    definition = []
    for word in sentence.split():
        definition.extend(tokenizer.tokenize(word))
    definition = tokenizer.convert_tokens_to_ids(definition)
    input = [cls_id] + [mask_id] * max_word_len + \
            [sep_id] + definition
    input = input[:256]
    input.append(sep_id)
    input = torch.LongTensor(input).unsqueeze(1) #(T,1)
    sen_list.append(sentence)
    in_list.append(input)


i = 0
b_size = 30
trans_tgt = []
tgt = []
while i < len(in_list):
    #import pdb; pdb.set_trace()
    input = in_list[i:i+b_size]
    input = pad_sequence(input, batch_first=True,padding_value=pad_id).squeeze(dim=-1)
    out = model(input)['pred']
    tgt_inds = list(torch.argmax(out, dim=-1))
    tgt_words = [idx2word[ind] for ind in tgt_inds]
    trans_words = [lookup[word] for word in tgt_words]
    tgt.extend(tgt_words)
    trans_tgt.extend(trans_words)
    i += b_size

log = [list(z) for z in zip(sen_list, tgt, trans_tgt)]
df = pd.DataFrame(log)
fname = "predictions_{}.xlsx".format(model_sel)
df.to_excel(fname, sheet_name='Sheet1')


