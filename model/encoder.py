import torch.nn as nn
import torch.nn.functional as F
from .multi_head_attention import MultiHeadAttention
from .input_embedding import WordEmbedding, PositionEmbedding

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super(EncoderBlock, self).__init__()
        self.embed_dim = config.EMBED_DIM
        self.hidden_dim = config.HIDDEN_DIM
        self.mha = MultiHeadAttention(config)
        self.linear1 = nn.Linear(self.embed_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.embed_dim)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
    
    def forward(self, x, seq_mask):
        output = self.mha(x, x, x, seq_mask) + x
        x = self.layer_norm1(output)
        output = self.linear2(F.relu(self.linear1(x))) + x
        output = self.layer_norm2(output)
        return output

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.num_blocks = config.ENCODER_N_BLOCKS
        self.embedding = WordEmbedding(config, type="en")
        self.position_embedding = PositionEmbedding(config)
        self.encoder = nn.ModuleList([
            EncoderBlock(config) for _ in range(self.num_blocks)
        ])
    
    def forward(self, x, seq_mask):
        output = self.embedding(x.int()) + self.position_embedding(x)
        for layer in self.encoder:
            output = layer(output, seq_mask)
        return output