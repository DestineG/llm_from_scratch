# src/test/token_embedding.py

import torch
import torch.nn.functional as F

from src.model.gpt import GPT
from src.tokenizer.bpe import build_bpe_tokenizer
from src.train.trainer import model_config

# -------------------------------
# 配置
# -------------------------------
pretrained_model_path = "experiments/gpt_owt_en_bpeTokenizer_with_warmup_v2/checkpoints/model_step_1340000.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 初始化 tokenizer 和模型
# -------------------------------
tokenizer = build_bpe_tokenizer(model_name="gpt2")
model_config.update({"vocab_size": tokenizer.n_vocab})
model = GPT(**model_config).to(device)
model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
embedding_layer = model.embedding.token_embedding
embedding_layer.eval()

# -------------------------------
# 测试词列表
# -------------------------------
words = [
    "king", "queen", "man", "woman", "dog", "cat", "apple", "orange"
]

# -------------------------------
# 获取 token ids
# -------------------------------
# GPT-2 tokenizer 可能会拆词，需要处理多 token
def get_word_embedding(word):
    tokens = tokenizer.encode(word)
    token_ids = torch.tensor(tokens, device=device)
    with torch.no_grad():
        embeds = embedding_layer(token_ids)  # [num_tokens, embed_dim]
        word_embed = embeds.mean(dim=0)      # 多 token 取平均
    return word_embed

embeddings = {w: get_word_embedding(w) for w in words}

# -------------------------------
# 计算余弦相似度
# -------------------------------
def cosine_similarity(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

print("Cosine similarity matrix:")
for i, w1 in enumerate(words):
    row = []
    for j, w2 in enumerate(words):
        sim = cosine_similarity(embeddings[w1], embeddings[w2])
        row.append(f"{sim:.2f}")
    print(f"{w1:>6}: {row}")
