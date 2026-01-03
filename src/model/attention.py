# src/model/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 将隐藏层维度划分为多个注意力头 不能整除就报错
        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(0.1)
        self.proj_dropout = nn.Dropout(0.1)
    
    # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
    def forward(self, x):
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, 3 * embed_dim)
        qkv = self.qkv_proj(x)
        batch_size, seq_len, _ = qkv.size()
        # (batch_size, seq_len, 3 * embed_dim) -> (batch_size, seq_len, 3, num_heads, head_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        # (batch_size, seq_len, 3, num_heads, head_dim) -> (3, batch_size, num_heads, seq_len, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # (3, batch_size, num_heads, seq_len, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # (batch_size, num_heads, seq_len, head_dim) * (batch_size, num_heads, head_dim, seq_len)
        # -> (batch_size, num_heads, seq_len, seq_len)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(scores.device)
        masked_scores = scores.masked_fill(mask == 0, float("-inf"))
        # (batch_size, num_heads, seq_len, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
        attn_weights = self.attn_dropout(F.softmax(masked_scores, dim=-1))
        # (batch_size, num_heads, seq_len, seq_len) * (batch_size, num_heads, seq_len, head_dim)
        # -> (batch_size, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, values)
        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, num_heads, head_dim)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, seq_len, embed_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        output = self.proj_dropout(self.out_proj(attn_output))
        return output
