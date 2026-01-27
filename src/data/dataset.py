# src/data/dataset.py

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from src.tokenizer.basic import build_tokenizer_from_wmt_en
from src.tokenizer.bpe import build_bpe_tokenizer
from .rawdata_handler import iter_wmt_en, iter_owt_en

class BasicDataset(Dataset):
    def __init__(self, window_size: int = 8, tokenizer=None, text_list=None):
        self.tokenizer = tokenizer
        self.text_list = text_list
        self.window_size = window_size
        
        self.data_ids = self.build_data()

    def build_data(self):
        all_ids = []
        eot_id = self.tokenizer.eot_token  # GPT-2 的结束符 ID

        for text in tqdm(self.text_list, desc="Encoding texts", ascii=True):
            all_ids.extend(self.tokenizer.encode(text))
            all_ids.append(eot_id)

        return all_ids

    def __len__(self):
        # 可取的样本数 = (总长度 - 窗口所需长度) // 步长
        # 预训练通常步长等于 window_size (不重叠)
        if len(self.data_ids) <= self.window_size:
            return 0
        return (len(self.data_ids) - 1) // self.window_size

    def __getitem__(self, idx):
        start = idx * self.window_size
        end = start + self.window_size + 1

        chunk = self.data_ids[start:end]

        # 不够长，跳过
        if len(chunk) < self.window_size + 1:
            return self.__getitem__((idx + 1) % len(self))

        eot = self.tokenizer.eot_token

        # 如果在输入区间内出现 EOT，丢弃这个样本
        if eot in chunk[:-1]:
            return self.__getitem__((idx + 1) % len(self))

        x = chunk[:-1]
        y = chunk[1:]

        return torch.tensor(x), torch.tensor(y)


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

def build_dataloader_from_owt_en_bpeTokenizer(
        text_nums: int = 10000, window_size: int = 8, batch_size: int = 32,
        num_workers: int = 0, shuffle: bool = True, bpe_model_name: str = "gpt2"):
    tokenizer = build_bpe_tokenizer(model_name=bpe_model_name)
    tokenizer.customName = "bpe_" + bpe_model_name
    text_list = list(iter_owt_en(text_nums))
    dataset = BasicDataset(window_size=window_size, tokenizer=tokenizer, text_list=text_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader, tokenizer

def test():
    # # 测试 BasicDataloader
    # dataloader, tokenizer = build_dataloader_from_wmt_en_basicTokenizer(
    #     seq_nums=1000, vocab_size=1000, window_size=16,
    #     batch_size=4, num_workers=0, shuffle=False
    # )
    # print(f"Vocabulary Size: {tokenizer.n_vocab}")
    # for i, (inputs, targets) in enumerate(dataloader):
    #     if i >= 1:
    #         break
    #     print(f"Batch {i}:")
    #     print("Inputs:", inputs.shape)
    #     print("Targets:", targets.shape)
    
    # 测试 BPE Dataloader
    dataloader_bpe, tokenizer_bpe = build_dataloader_from_wmt_en_bpeTokenizer(
        seq_nums=1000, window_size=16,
        batch_size=4, num_workers=0, shuffle=False,
        bpe_model_name="gpt2"
    )
    print(f"\nlength of set: {len(dataloader_bpe.dataset)}")
    print(f"\nBPE Vocabulary Size: {tokenizer_bpe.n_vocab}")
    for i, (inputs, targets) in enumerate(dataloader_bpe):
        if i >= 1:
            break
        print(f"Batch {i}:")
        print("Inputs:", inputs.shape)
        print("Targets:", targets.shape)
    
    # 测试 IMDB
    dataloader_bpe, tokenizer_bpe = build_imdb_classification_dataloader(
        batch_size=32, max_len=128,
        num_workers=4, shuffle=True, split="train"
    )
    print(f"\nlength of train set: {len(dataloader_bpe.dataset)}")
    print(f"\nBPE Vocabulary Size: {tokenizer_bpe.n_vocab}")
    for i, (inputs, targets) in enumerate(dataloader_bpe):
        if i >= 1:
            break
        print(f"Batch {i}:")
        print("Inputs:", inputs.shape)
        print("Targets:", targets.shape)

    dataloader_bpe, tokenizer_bpe = build_imdb_classification_dataloader(
        batch_size=32, max_len=128,
        num_workers=4, shuffle=False, split="test"
    )
    print(f"\nlength of test set: {len(dataloader_bpe.dataset)}")
    print(f"\nBPE Vocabulary Size: {tokenizer_bpe.n_vocab}")
    for i, (inputs, targets) in enumerate(dataloader_bpe):
        if i >= 1:
            break
        print(f"Batch {i}:")
        print("Inputs:", inputs.shape)
        print("Targets:", targets.shape)


class IMDBClassificationDataset(Dataset):
    def __init__(self, tokenizer, split="train", max_len=128):
        self.tokenizer = tokenizer
        self.max_len = max_len

        print(f"Loading IMDb {split} dataset...")
        data_dir = 'data/raw/imdb'
        data_files = {
            'train': f'{data_dir}/train-00000-of-00001.parquet',
            'test': f'{data_dir}/test-00000-of-00001.parquet'
        }
        self.dataset = load_dataset("parquet", data_files=data_files)[split]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        label = item['label']

        # 编码文本
        ids = self.tokenizer.encode(text)
        
        # 截断
        if len(ids) > self.max_len:
            ids = ids[:self.max_len]
        
        # 左填充 (Left Padding)
        # 注意：GPT2 默认没有专门的 pad_token，通常复用 eot_token
        pad_id = self.tokenizer.eot_token 
        padding_len = self.max_len - len(ids)
        ids = [pad_id] * padding_len + ids

        return torch.tensor(ids), torch.tensor(label)

def build_imdb_classification_dataloader(
        batch_size: int = 32, max_len: int = 128, num_workers: int = 0,
        shuffle: bool = True, split: str = "train", bpe_model_name: str = "gpt2"):
    
    # 复用你现有的 BPE Tokenizer 加载逻辑
    tokenizer = build_bpe_tokenizer(model_name=bpe_model_name)
    tokenizer.customName = "bpe_" + bpe_model_name + "_imdb"
    
    dataset = IMDBClassificationDataset(tokenizer=tokenizer, split=split, max_len=max_len)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers
    )
    return dataloader, tokenizer

if __name__ == "__main__":
    test()