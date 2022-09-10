import torch
import torch.nn as nn
import math

class WordEmbedding(nn.Module):
    def __init__(self, config, type="en"):
        super(WordEmbedding, self).__init__()
        if type == "en":
            self.vocab_size = config.ENG_VOCAB_SIZE
        else:
            self.vocab_size = config.FR_VOCAB_SIZE
        self.emb = nn.Embedding(self.vocab_size, config.EMBED_DIM)
        self.to(config.DEVICE)
    def forward(self, x):
        out = self.emb(x)
        return out

class PositionEmbedding:
    def __init__(self, config):
        self.max_seq_len = config.MAX_SEQ_LEN
        self.embed_dim = config.EMBED_DIM
        pe = torch.zeros((self.max_seq_len, self.embed_dim))
        for pos in range(self.max_seq_len):
            for i in range(self.embed_dim):
                if i % 2 == 0:
                    pe[pos][i] = math.sin(pos/(10000**(i/self.embed_dim)))
                else:
                    pe[pos][i] = math.cos(pos/(10000**((i-1)/self.embed_dim)))
        self.pe = pe.unsqueeze(0)
        self.device = config.DEVICE
    def __call__(self, x):
        # x = x * math.sqrt(self.embed_dim)
        seq_len = x.size(1)
        return self.pe[:,:seq_len,:].to(self.device)
