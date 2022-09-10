import torch.nn as nn
import torch.nn.functional as F
from .multi_head_attention import MultiHeadAttention
from .input_embedding import WordEmbedding, PositionEmbedding

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super(DecoderBlock, self).__init__()
        self.embed_dim = config.EMBED_DIM
        self.hidden_dim = config.HIDDEN_DIM
        self.masked_mha = MultiHeadAttention(config)
        self.mha = MultiHeadAttention(config)
        self.layernorm1 = nn.LayerNorm(self.embed_dim)
        self.layernorm2 = nn.LayerNorm(self.embed_dim)
        self.layernorm3 = nn.LayerNorm(self.embed_dim)
        self.feedfw = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.embed_dim)
        )

    def forward(self, x, encoder_output, trg_seq_mask, seq_mask):
        output = self.masked_mha(x, x, x, trg_seq_mask) + x
        x = self.layernorm1(output)
        output = self.mha(x, encoder_output, encoder_output, seq_mask) + x
        x = self.layernorm2(output)
        output = self.feedfw(x) + x
        output = self.layernorm3(output)
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, config):
        super(TransformerDecoder, self).__init__()
        self.embed_dim = config.EMBED_DIM
        self.num_blocks = config.DECODER_N_BLOCKS
        self.embedding = WordEmbedding(config, type="fr")
        self.position_embedding = PositionEmbedding(config)
        self.decoder = nn.ModuleList([
            DecoderBlock(config) for _ in range(self.num_blocks)
        ])
        self.linear = nn.Linear(self.embed_dim, self.embedding.vocab_size)
    def forward(self, encoder_output, x, trg_seq_mask, seq_mask):
        output = self.embedding(x.int()) + self.position_embedding(x)
        for layer in self.decoder:
            output = layer(output, encoder_output, trg_seq_mask, seq_mask)
        return self.linear(output)