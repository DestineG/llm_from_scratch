# src/model/embeddings.py

import torch
import torch.nn as nn

class BasicEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, seq_len):
        super(BasicEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(seq_len, embed_dim)

    # (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
    def forward(self, input_ids):
        # (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        token_embeddings = self.token_embedding(input_ids)

        batch_size, seq_len = input_ids.size()
        # (seq_len,)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        # (seq_len,) -> (1, seq_len) -> (batch_size, seq_len)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        # (batch_size, seq_len) -> (batch_size, seq_len, embed_dim)
        position_embeddings = self.position_embedding(position_ids)
        
        return token_embeddings + position_embeddings


if __name__ == "__main__":
    # Test the BasicEmbedding module
    vocab_size = 100
    seq_len = 10
    embed_dim = 32
    batch_size = 2

    model = BasicEmbedding(vocab_size, seq_len, embed_dim)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    embeddings = model(input_ids)
    print("Input IDs:", input_ids)
    print("Embeddings shape:", embeddings.shape)  # Expected shape: (batch_size, seq_len, embed_dim)