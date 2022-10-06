import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence


NEG_INF = -100000000.0 

def mask_softmax(e, lengths):
    """Returns masked softmax"""
    """e --> (N, L) N = Number of sentences in Batch, L = Length of longest sequence
        lengths of each sequeunce --> (N,)"""
    lengths = lengths.reshape(-1,1) #(N,1)
    N = lengths.shape[0]
    len_max = torch.max(lengths).item()
    L = (torch.arange(len_max).repeat(N,1)).to(lengths) #(N, len_max)
    ones = torch.ones(N, len_max).to(L)
    zeros = torch.zeros(N, len_max).to(L)
    mask = torch.where(lengths > L, zeros, ones) #(N, len_max)
    return  F.softmax(e + NEG_INF*mask, dim=-1) #(N, len_max)




class Net(nn.Module):
    def __init__(self, dim, num_outclass, num_inclass, embed_dim=20):
        super(Net, self).__init__()
        
        self.dim = dim

        self.enc = nn.LSTM(input_size = embed_dim, hidden_size = dim, batch_first=True, num_layers=2, bidirectional=True)
        self.embed = nn.Embedding(num_embeddings=num_inclass, embedding_dim=embed_dim)

        self.W_s = nn.Linear(in_features=2*dim, out_features=2*dim)
        self.dec = nn.LSTM(input_size = 2*dim, hidden_size = 2*dim, batch_first=True, num_layers=1, bidirectional=False) #Single stage every time
        
        #Attention Layer Components
        self.W_a = nn.Linear(in_features=2*dim, out_features=2*dim)
        self.U_a = nn.Linear(in_features=2*dim, out_features=2*dim)
        self.v_a = nn.Linear(in_features=2*dim, out_features=1)

        #Linear layer to calculate Scores for Classification
        self.cls = nn.Linear(in_features=2*dim, out_features=num_outclass)
     
    def forward(self, X, seq_len):

        hid,_ =  self.enc(self.embed(X)) #(N, len_max, 2d)
        
        tanh = nn.Tanh()
        #len_max = torch.max(seq_len)
        N = seq_len.shape[0]
        D = hid.shape[2] #D == 2d
        s_i = tanh(self.W_s(hid[:,0])) #(N,2d)
        l = []
        for _ in range(0, 11):
            z_i = torch.zeros(1,N,D).to(hid) #(1,N,2d) num_layers*num_direction = 1 
            s_i = s_i.unsqueeze(dim=1) #(N,1,2d)
            v1 = self.W_a(s_i) #(N,1,2d)
            v2 = self.U_a(hid) #(N, len_max, 2d)
            e = self.v_a(tanh(v1 + v2)).squeeze(dim=-1) #(N, len_max, 1) --> (N, len_max)
            att = mask_softmax(e, seq_len) #(N, len_max)
            C = att.unsqueeze(dim=-1)*hid #(N, len_max, 2d)
            ci = torch.sum(C, dim=1) #(N, 2d) -- context vector for this time in the time sequence
            s_i = s_i.squeeze(dim=1) #(N,2d)
            s_i = s_i.unsqueeze(dim=0)#(1,N,2d) num_layers*num_direction = 1 
            s_i, _ = self.dec(ci.unsqueeze(dim=1), (s_i,z_i)) #(N,1,2d)
            s_i = s_i.squeeze(dim=1)
            l.append(s_i)

        s = (torch.stack(l, dim=1)).to(hid) #(N, 11, 2d) ,11 for the END token
        scores = self.cls(s) #(N, 11, num_outclass)
        
        return scores

    def predict(self, X, lens):
        scores = self(X, lens) #(N, 11, num_outclass)
        return torch.argmax(scores, dim=-1) #(N, 11)
    
