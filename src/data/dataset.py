# src/data/dataset.py

import torch
from torch.utils.data import Dataset, DataLoader

from src.tokenizer.basic import build_tokenizer_from_wmt_en
from src.tokenizer.bpe import build_bpe_tokenizer
from .rawdata_handler import iter_wmt_en

class BasicDataset(Dataset):
    def __init__(self, window_size: int = 8, tokenizer=None, text_list=None):
        self.tokenizer = tokenizer
        self.text_list = text_list
        self.window_size = window_size
        
        self.data_ids = self.build_data()

    def build_data(self):
        all_ids = []
        for text in self.text_list:
            all_ids.extend(self.tokenizer.encode(text))
            if hasattr(self.tokenizer, "eot_token"):
                all_ids.append(self.tokenizer.eot_token)
            else:
                all_ids.extend(self.tokenizer.encode("<eos>"))
        return all_ids

    def __len__(self):
        # 可取的样本数 = (总长度 - 窗口所需长度) // 步长
        # 预训练通常步长等于 window_size (不重叠)
        if len(self.data_ids) <= self.window_size:
            return 0
        return (len(self.data_ids) - 1) // self.window_size

    def __getitem__(self, idx):
        # 计算当前 chunk 的起始位置
        start = idx * self.window_size
        end = start + self.window_size
        
        # 构造输入和目标
        # Input:  [0, 1, 2, 3]
        # Target: [1, 2, 3, 4] (预测下一个 token)
        input_seq = self.data_ids[start : end]
        target_seq = self.data_ids[start + 1 : end + 1]
        
        return torch.tensor(input_seq), torch.tensor(target_seq)

    def debug_sample(self, idx):
        """垂直对齐打印调试信息"""
        x, y = self.__getitem__(idx)
        print(f"=== Sample Index: {idx} ===")
        print(f"{'Input IDs:':<15} {x.tolist()}")
        print(f"{'Target IDs:':<15} {y.tolist()}")
        # 还原回文本看一眼语义
        print(f"{'Input Text:':<15} {self.tokenizer.detokenize(x.tolist())}")
        print("-" * 50)

def build_dataloader_from_wmt_en_basicTokenizer(
        seq_nums: int = 10000, vocab_size: int = 20000, window_size: int = 8,
        batch_size: int = 32, num_workers: int = 0, shuffle: bool = True):
    tokenizer = build_tokenizer_from_wmt_en(limit=seq_nums, max_size=vocab_size)
    tokenizer.customName = f"basic_{vocab_size}"
    text_list = list(iter_wmt_en(seq_nums))
    dataset = BasicDataset(window_size=window_size, tokenizer=tokenizer, text_list=text_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader, tokenizer

def build_dataloader_from_wmt_en_bpeTokenizer(
        seq_nums: int = 10000, window_size: int = 8, batch_size: int = 32,
        num_workers: int = 0, shuffle: bool = True, bpe_model_name: str = "gpt2"):
    tokenizer = build_bpe_tokenizer(model_name=bpe_model_name)
    tokenizer.customName = "bpe_" + bpe_model_name
    text_list = list(iter_wmt_en(seq_nums))
    dataset = BasicDataset(window_size=window_size, tokenizer=tokenizer, text_list=text_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader, tokenizer

def test():
    # 测试 BasicDataloader
    dataloader, tokenizer = build_dataloader_from_wmt_en_basicTokenizer(
        seq_nums=1000, vocab_size=1000, window_size=16,
        batch_size=4, num_workers=0, shuffle=False
    )
    print(f"Vocabulary Size: {tokenizer.n_vocab}")
    for i, (inputs, targets) in enumerate(dataloader):
        print(f"Batch {i}:")
        print("Inputs:", inputs)
        print("Targets:", targets)
        if i >= 2:
            break
    
    # 测试 BPE Dataloader
    dataloader_bpe, tokenizer_bpe = build_dataloader_from_wmt_en_bpeTokenizer(
        seq_nums=1000, window_size=16,
        batch_size=4, num_workers=0, shuffle=False,
        bpe_model_name="gpt2"
    )
    print(f"\nBPE Vocabulary Size: {tokenizer_bpe.n_vocab}")
    for i, (inputs, targets) in enumerate(dataloader_bpe):
        print(f"Batch {i}:")
        print("Inputs:", inputs)
        print("Targets:", targets)
        if i >= 2:
            break


if __name__ == "__main__":
    test()