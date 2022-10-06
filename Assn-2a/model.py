import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence

class Net(nn.Module):
    def __init__(self,dim):
        super(Net, self).__init__()
        self.l0 = nn.LSTM(input_size = dim,hidden_size = dim, batch_first = True, num_layers=2)
        self.l1 = nn.Linear(dim, 2)
        
    def forward(self,X):
        # Both of them are equal (X[0].batch_sizes) = (pre.batch_sizes)
        time_dimension = 1
        
        #premise/input
        hid,_ = self.l0(X)
        hid_unp, hid_lens = pad_packed_sequence(hid, batch_first=True) #hid_unp -> unpacked paddings, hid_lens -> time-length of each sentence in the batch
        idx = torch.LongTensor(hid_lens)-1 #Getting the last time stamp for each sample in the batch
        idx = idx.view(-1,1).expand(len(hid_lens),hid_unp.size(2)) #hid_unp.size(2) = dim
        idx = idx.unsqueeze(time_dimension)
        if hid_unp.is_cuda:
            idx = idx.cuda(hid_unp.data.get_device())
        
        hid =  hid_unp.gather(time_dimension,idx).squeeze(time_dimension)
        return self.l1(hid)
    
    def predict(self,X):
        out = self.forward(X)
        prediction = torch.argmax(out,dim=1)
        return prediction