import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence

NEG_INF = -1000000

def mask_softmax(matrix, mask=None):
    """Perform softmax on length dimension with masking.

    Parameters
    ----------
    matrix: torch.float, shape [batch_size, .., max_len]
    mask: torch.long, shape [batch_size, max_len]
        Mask tensor for sequence.

    Returns
    -------
    output: torch.float, shape [batch_size, .., max_len]
        Normalized output in length dimension.
    """

    if mask is None:
        result = F.softmax(matrix, dim=-1)
    else:
        mask_norm = (mask * NEG_INF).to(matrix)
        result = F.softmax(matrix + mask_norm, dim=-1)

    return result

def seq_mask(seq_len, max_len):
    """Create sequence mask.

    Parameters
    ----------
    seq_len: torch.long, shape [batch_size],
        Lengths of sequences in a batch.
    max_len: int
        The maximum sequence length in a batch.

    Returns
    -------
    mask: torch.long, shape [batch_size, max_len]
        Mask tensor for sequence.
    """

    idx = torch.arange(max_len).to(seq_len).repeat(seq_len.size(0), 1)
    mask = torch.gt(seq_len.unsqueeze(1), idx).to(seq_len)
    mask = 1-mask
    return mask


class Net(nn.Module):
    def __init__(self,dim):
        super(Net, self).__init__()
        self.l0 = nn.GRU(input_size = dim,hidden_size = dim, batch_first = True, num_layers=2, bidirectional=True)
        self.att = nn.Linear(2*dim, 1) #Attention layer
        self.l1 = nn.Linear(2*dim, 2)
        
    def forward(self,X):
        # Both of them are equal (X[0].batch_sizes) = (pre.batch_sizes)
        time_dimension = 1
        
        #premise/input
        hid,_ = self.l0(X)
        hid_unp, seq_len = pad_packed_sequence(hid, batch_first=True) #hid_unp -> unpacked paddings, hid_lens -> time-length of each sentence in the batch
        
        #Attention computation
        max_seq_len = torch.max(seq_len)
        mask = seq_mask(seq_len, max_seq_len) #[batch_size, max_len]
        att = self.att(hid_unp).squeeze(-1) #[batch_size, max_len]
        att = mask_softmax(att, mask)

        r_att = torch.sum(att.unsqueeze(-1) * hid_unp, dim=1) #[batch_size, D*embed_dim], D=2(since bi-directional)att
        return self.l1(r_att)
    
    def predict(self,X):
        out = self.forward(X)
        prediction = torch.argmax(out,dim=1)
        return prediction