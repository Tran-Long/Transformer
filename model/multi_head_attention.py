import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = config.EMBED_DIM
        self.n_heads = config.N_HEADS
        self.single_head_dim = int(self.embed_dim/self.n_heads)
        self.d_k, self.d_v = self.single_head_dim, self.single_head_dim
        self.q_matrix = nn.Linear(self.embed_dim, self.d_k * self.n_heads)
        self.k_matrix = nn.Linear(self.embed_dim, self.d_k * self.n_heads)
        self.v_matrix = nn.Linear(self.embed_dim, self.d_v * self.n_heads)
        self.out_layer = nn.Linear(self.n_heads * self.d_v, self.embed_dim)

    
    def forward(self, Q, K, V, mask=None):
        """
            Q, K, V: (B, L, embed_dim). L: len sentence
        """
        batch_size, sen_len, embed_dim = Q.shape
        q = self.q_matrix(Q).transpose(-2, -1).view(batch_size, self.n_heads, self.d_k, sen_len).transpose(-2, -1)
        k = self.k_matrix(K).transpose(-2, -1).view(batch_size, self.n_heads, self.d_k, -1).transpose(-2, -1)
        v = self.v_matrix(V).transpose(-2, -1).view(batch_size, self.n_heads, self.d_v, -1).transpose(-2, -1)

        score = torch.matmul(q, k.transpose(-2, -1)) # (B, n_heads, sen_len/target_sen_len, sen_len)
        score /= np.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
            score = score.masked_fill(mask == 0, float("-1e20"))
        score = F.softmax(score, dim=-1)
        z = torch.matmul(score, v) # (B, n_heads, sen_len/target_sen_len, d_v)
        return z.transpose(-2, -3).reshape(batch_size, sen_len, embed_dim) # (B, sen_len/target_sen_len, d_models)
