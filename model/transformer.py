import torch
import torch.nn as nn
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        self.device = config.DEVICE
        self.to(config.DEVICE)

    def make_mask(self, seq_pad_mask, tar_seq, trg_seq_pad_mask):
        batch_size, max_seq_len = tar_seq.shape
        mask = torch.tril(torch.ones((max_seq_len, max_seq_len))).expand(
            batch_size, max_seq_len, max_seq_len
        ).to(self.device)
        seq_pad_mask = seq_pad_mask.unsqueeze(1).expand(batch_size, max_seq_len, max_seq_len)
        trg_seq_pad_mask = trg_seq_pad_mask.unsqueeze(1).expand(batch_size, max_seq_len, max_seq_len)
        trg_seq_pad_mask = trg_seq_pad_mask * mask
        return seq_pad_mask, trg_seq_pad_mask

    def forward(self, seq, seq_pad_mask, tar_seq, trg_seq_pad_mask):
        seq_mask, trg_seq_mask = self.make_mask(seq_pad_mask, tar_seq, trg_seq_pad_mask)
        encoder_output = self.encoder(seq, seq_mask)
        out = self.decoder(encoder_output, tar_seq, trg_seq_mask, seq_mask)
        return out