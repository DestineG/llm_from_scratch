# src/model/transformer.py

import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadAttention

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
    
    # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
    def forward(self, x):
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, ff_dim)
        x = self.dropout(F.gelu(self.fc1(x)))
        # (batch_size, seq_len, ff_dim) -> (batch_size, seq_len, embed_dim)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, ff_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
    
    # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
    def forward(self, x):
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        attn_out = self.self_attn(self.norm1(x))
        x = x + self.dropout(attn_out)

        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)

        return x