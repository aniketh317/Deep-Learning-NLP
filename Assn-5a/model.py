import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, edim=300, n_heads=6, num_layers=4):
        assert edim % n_heads ==0

        """
        edim --> Dimension of the embeddings
        n_heads --> Number of heads in the multi-dimesion embedding
        num_layers --> Number of sub-encoder layers stacked up to form the encoder
        """
        dff = edim*4 #Dimension of FeedForward (edim -> dff -> edim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=edim, nhead=n_heads, dim_feedforward=dff, batch_first=True) #Sub-encoder layer
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.last = nn.Sequential(nn.ReLU(), nn.Linear(in_features=edim, out_features=2))

    def forward(self, src, mask=None):
        scores =  self.last(self.encoder(src=src, mask=mask))
        return scores #(N, 2)
    
    def predict(self, src, mask=None):
        scores = self.forward(src, mask) #(N, 2)
        return torch.argmax(scores, dim=-1)