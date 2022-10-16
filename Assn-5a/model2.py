import torch
import torch.nn as nn
import transformers

class Net(nn.Module):
    def __init__ (self):
        super(Net, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.bert_drop = nn.Dropout(0.2)
        self.out = nn.Linear(768, 2) #Since output is 2 class
        
    def forward(self, src, src_mask, token_type_ids=None):
        _, pooledOut = self.bert(src, attention_mask = src_mask,
                                token_type_ids=token_type_ids,return_dict=False)
        bertOut = self.bert_drop(pooledOut)
        output = self.out(bertOut)
        
        return output
    
    def predict(self, src, src_mask, token_type_ids=None):
        scores = self.forward(src, src_mask, token_type_ids) #(N, 2)
        return torch.argmax(scores, dim=-1) #(N,)