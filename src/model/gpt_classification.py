# src/model/gpt_classification

import torch.nn as nn

class GPTForClassification(nn.Module):
    def __init__(self, pretrained_gpt, num_classes):
        super().__init__()
        self.gpt = pretrained_gpt
        self.embed_dim = pretrained_gpt.embed_dim
        # 分类线性层
        self.classifier = nn.Linear(self.embed_dim, num_classes)
        
    def forward(self, input_ids):
        # GPT 输出: (batch, seq_len, embed_dim)
        hidden_states = self.gpt(input_ids, return_hidden=True)
        
        # 取最后一个有效 Token 的状态 (针对分类任务的通用做法)
        # 如果是 Left Padding，直接取最后一个 index 即可
        last_hidden_state = hidden_states[:, -1, :] 
        
        logits = self.classifier(last_hidden_state)
        return logits