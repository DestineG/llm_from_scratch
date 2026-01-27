# src/model/gpt.py

import torch
import torch.nn as nn
from .embeddings import BasicEmbedding
from .transformer import TransformerBlock

class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, seq_len, num_heads, ff_dim, num_layers):
        super(GPT, self).__init__()

        self.embed_dim = embed_dim
        
        # 词向量与位置向量
        self.embedding = BasicEmbedding(vocab_size, embed_dim, seq_len)
        
        # Transformer
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim) 
            for _ in range(num_layers)
        ])
        
        # 最后的 LayerNorm
        self.ln_f = nn.LayerNorm(embed_dim)
        
        # token 预测头
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # 权重共享
        self.lm_head.weight = self.embedding.token_embedding.weight
        
        # 权重初始化逻辑
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # (batch_size, seq_len) -> (batch_size, seq_len, vocab_size)
    def forward(self, idx, return_hidden=False):
        # (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        x = self.embedding(idx)
        
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        for block in self.blocks:
            x = block(x)
        
        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, embed_dim)
        x = self.ln_f(x)

        if return_hidden:
                return x  # 直接返回 (batch, seq_len, 768)

        # (batch_size, seq_len, embed_dim) -> (batch_size, seq_len, vocab_size)
        logits = self.lm_head(x)
        
        return logits


if __name__ == "__main__":
    model = GPT(vocab_size=30522, embed_dim=768, seq_len=128, num_heads=12, ff_dim=3072, num_layers=12)
    input_ids = torch.randint(0, 30522, (2, 128))
    outputs = model(input_ids)
    print(outputs.shape)