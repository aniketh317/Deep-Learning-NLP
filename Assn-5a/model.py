import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

NEG_INF = -1000000000.0 

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout=0.1, max_len=3000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)].to(x), requires_grad=False)
        return self.dropout.to(x)(x)


class Net(nn.Module):
    def __init__(self, vocab=None, edim=None, init_embedding=None, n_heads=6, num_layers=2):

        assert (init_embedding is not None) or (vocab is not None and edim is not None) #Ensure that atleast one of them is supplied not None

        """
        vocab --> Length of the vocabulary (This also includes the one for pad token)
        edim --> Dimension of the embeddings
        n_heads --> Number of heads in the multi-dimesion embedding
        num_layers --> Number of sub-encoder layers stacked up to form the encoder
        """
        super(Net, self).__init__()
        self.embed = None
        if init_embedding is not None:
            vocab, edim = init_embedding.shape
            self.embed = nn.Embedding.from_pretrained(init_embedding, freeze=False, padding_idx=vocab-1) #The embeddings are learnable
        else:    
            self.embed = nn.Embedding(num_embeddings=vocab, embedding_dim=edim, padding_idx=vocab-1) #The embeddings are frozen

        assert edim % n_heads == 0 #Embedding dimension to be an integral multiple of num_heads in the Multi Head Attention

        dff = edim*4 #Dimension of FeedForward (edim -> dff -> edim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=edim, nhead=n_heads, dim_feedforward=dff, batch_first=True) #Sub-encoder layer
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        
        self.qa = nn.Linear(in_features=edim, out_features=1) #The single query term for which is important among the representation of the T seq. elements
        self.last = nn.Sequential(nn.ReLU(), nn.Linear(in_features=edim, out_features=2))
        self.posencoder = PositionalEncoding(d_model=edim, dropout=0.1)

        params = list(self.encoder.parameters())+list(self.qa.parameters())+list(self.last.parameters())
        for p in params:
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
     

    def forward(self, src, src_mask=None):
        """
        src --> Source (N, T)
        pad_mask --> (N, T), pad_mask[n][i] = True, then ith term in nth data in batch is PAD/UNK.
        """
        emb = self.posencoder(self.embed(src)) #(N, T, edim)

        eout =  self.encoder(src=emb, src_key_padding_mask=src_mask) #Encoder output (N, T, edim)
        atts = self.qa(eout) #(N, T, 1)
        atts = atts.squeeze(dim=-1) #(N, T)
        if src_mask is not None:
            atts = atts.masked_fill(src_mask, NEG_INF) #(N, T)
        atts = F.softmax(atts, dim=-1) #(N, T)

        atts = atts.unsqueeze(dim=-2) #(N, 1, T)
        out = torch.matmul(atts, eout) #(N, 1, edim)
        out = out.squeeze(dim=-2) #(N, edim)
        scores = self.last(out) #(N, 2)
        return scores #(N, 2)
    
    def predict(self, src, src_mask=None):
        scores = self.forward(src, src_mask) #(N, 2)
        return torch.argmax(scores, dim=-1)