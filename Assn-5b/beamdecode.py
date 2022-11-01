import torch
import torch.nn as nn
from queue import PriorityQueue

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

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

class BeamSearchNode(object):
    def __init__(self, decode, wordId, logProb, length):
        '''
        :param decode :The decode input for next round(The last word concatendated)
        :param wordId: The id of the last word
        :param logProb:
        :param length:
        '''
        self.decode = decode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self):
        # Add here a function for shaping a reward

        return self.logp

def beam_decode(model, src, src_mask, max_len, start_symbol, beam=3):
    '''
    :param beam: size of the beam
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''

    beam_width = beam
    topk = 1  # how many sentence do you want to generate

    # decoding goes sentence by sentence
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    memory = model.encode(src, src_mask) #(S, E)

    # Start with the start of the sentence token
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)

    # Number of sentence to generate
    endnodes = []
    number_required = min((topk + 1), topk - len(endnodes))

    # starting node -  hidden vector, previous node, word id, logp, length
    node = BeamSearchNode(ys, ys.item(),  0, 1)
    nodes = PriorityQueue()

    # start the queue
    nodes.put((-node.eval(), node))

    # start beam search
    while True:

        # fetch the best node
        score, n = nodes.get()
        ys = n.decode

        if n.wordid == EOS_IDX:
            endnodes.append((score, n))
            # if we reached maximum # of sentences required
            if len(endnodes) >= number_required:
                break
            else:
                continue

        # decode for one step using decoder
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask) #(T,1,E)
        out = out.transpose(0, 1) #(1, T, E)
        prob = model.generator(out[:, -1]) #(1, outvocab)
        log_pf = nn.LogSoftmax()(prob[0]) #(outvocab,)

        # PUT HERE REAL BEAM SEARCH OF TOP
        log_pt, indexes = torch.topk(log_pf, beam_width) #(beamwidth,) log_pt->log probabilities of top k
        nextnodes = []

        for new_k in range(beam_width):
            decoded_t = indexes[new_k].item() #Get the word-index for the respective sentence
            log_p = log_pt[new_k].item()
            ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(decoded_t)], dim=0)
            node = BeamSearchNode(ys, decoded_t, n.logp + log_p, n.leng + 1)
            score = -node.eval()
            nextnodes.append((score, node))

        # put them into queue
        for i in range(len(nextnodes)):
            score, nnode = nextnodes[i]
            nodes.put((score, nnode))

    # choose nbest paths, back trace them
    if len(endnodes) == 0:
        endnodes = [nodes.get() for _ in range(topk)]

    _,n = endnodes[0]
    return n.decode