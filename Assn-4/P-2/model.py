import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = -100000000.0 

def mask_softmax(e, lengths):
    """Returns masked softmax"""
    """e --> (N, len_max) N = Number of datapoints(sequences) in Batch, len_max = Length of longest sequence
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
    def __init__(self, dim, embed_dim=100):
        super(Net, self).__init__()
        self.tanh = nn.Tanh()# Have a tanh layer initialized for all

        self.wenc = nn.GRU(input_size=embed_dim, hidden_size=dim, batch_first=True, num_layers=1, bidirectional=True) #Word-encoder
        self.Ww = nn.Linear(in_features=2*dim, out_features=dim) #This has a parameter for bias
        self.uw = nn.Linear(in_features = dim, out_features=1)

        self.senc = nn.GRU(input_size=2*dim, hidden_size=dim, batch_first=True, num_layers=1, bidirectional=True) #Sentence-encoder
        self.Ws = nn.Linear(in_features=2*dim, out_features=dim) #This has a parameter for bias
        self.us = nn.Linear(in_features=dim, out_features=1) #This has a parameter for bias 
        
        self.last = nn.Linear(in_features=2*dim, out_features=2) #Binary classification problem, hence 2. This has a parameter for bias
    
    def forward(self, X, lend, lens):
        """
        Input:
        X --(B, Ms, Mw, edim) : B-Batch size, Ms-Max number of sentences, Mw-Max number of words in a sentence(over all sentences), edim-embedding dimension
        lend -- (B, ) :Length of each document in batch(number of sentences in the document)
        lens-- (B, Ms): Length of each sentence in each document of the batch lens[b][m] is length of mth sentence of bth doc in batch  
        
        Returns:
        scores -- (B, 2) :Scores for class 0, class 1 (for every document)
        """
        # d = dim
        # ed = edim
        (B, Ms) = (X.shape[0], X.shape[1])

        """Word level in hierarchy"""
        Xw = torch.flatten(X, start_dim=0, end_dim=1) #(B*Ms, Mw, edim)
        lens = lens.reshape(-1) #(B*Ms, )
        Xw = nn.utils.rnn.pack_padded_sequence(Xw, lens.to(torch.device('cpu')),batch_first = True, enforce_sorted=False)
        hidw_pack,_ =  self.wenc(Xw) #(B*Ms, Mw, 2d) Output is pack_padded_sequence
        hidw,_ = nn.utils.rnn.pad_packed_sequence(hidw_pack, batch_first=True) #hidw->unpacked paddings, _->length of each sentence in the lot
        
        outw = self.tanh(self.Ww(hidw)) #(B*Ms, Mw, d)
        ew  = self.uw(outw) #(B*Ms, Mw, 1)
        ew = ew.squeeze(dim=-1) #(B*Ms, Mw)
        attw = mask_softmax(ew, lens) #(B*Ms, Mw)
        attw = attw.unsqueeze(dim=-1) #(B*Ms, Mw, 1)
        s = attw*hidw #(B*Ms, Mw, 2d)
        se = torch.sum(s, dim=1) #(B*Ms, 2d) -- Sentence encodings for every sentence in the batch
        
        """Sentence level in hierarchy"""
        Xs = torch.reshape(se, (B, Ms, -1)) #(B, Ms, 2d)
        Xs = nn.utils.rnn.pack_padded_sequence(Xs, lend.to(torch.device('cpu')),batch_first = True, enforce_sorted=False)
        hids_pack,_ = self.senc(Xs) #(B, Ms, 2d) Output is pack_padded_sequence
        hids,_ = nn.utils.rnn.pad_packed_sequence(hids_pack, batch_first=True) #hids->unpacked paddings, _->length of each doc in the batch

        outs = self.tanh(self.Ws(hids)) #(B, Ms, d)
        es = self.us(outs) #(B, Ms, 1)
        es = es.squeeze(dim=-1)
        atts = mask_softmax(es, lend) #(B, Ms)
        atts = atts.unsqueeze(dim=-1) #(B, Ms, 1)
        d = atts*hids #(B, Ms, 2d)
        de = torch.sum(d, dim=1) #(B, 2d) -- Document encodings for each documnet in the batch

        scores = self.last(de) #(B, 2)
        return scores

    def predict(self, X, lend, lens):
        """
        Input:
        X --(B, Ms, Mw, edim) : B-Batch size, Ms-Max number of sentences, Mw-Max number of words in a sentence(over all sentences), edim-embedding dimension
        lend -- (B, ) :Length of each document in batch(number of sentences in the document)
        lens-- (B, Ms): Length of each sentence in each document of the batch lens[b][m] is length of mth sentence of bth doc in batch

        Returns:
        Predictions for each doc  
        """
        scores = self.forward(X, lend, lens) #(B,2)
        return torch.argmax(scores, dim=-1)