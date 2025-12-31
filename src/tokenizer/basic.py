# src/tokenizer/basic.py

import re
import hashlib
from collections import Counter

from src.data.rawdata_handler import iter_wmt_en, iter_wmt_zh
from src.utils.cache import cacheExist, cacheLoad, cacheSave

class Vocab:
    def __init__(self,
                 counter,
                 max_size: int = -1,
                 min_freq: int = 1,
                 special_tokens=["<pad>", "<unk>"]
    ):
        """
        counter: 一个 Counter 对象，统计了 token 的频率
        special_tokens: 预定义的特殊 token 列表，默认包含 <pad> 和 <unk>
        max_size: 最大 vocab 大小（包含特殊 token），-1 表示无限制
        min_freq: 最小 token 出现频率，低于的 token 会被丢弃
        """
        self.special_tokens = special_tokens

        # 过滤掉低频 token
        filtered_tokens = [t for t, freq in counter.items() if freq >= min_freq]

        # 先加入特殊 token，再去重
        all_tokens = list(dict.fromkeys(self.special_tokens + filtered_tokens))

        # 截断到 max_size
        if max_size > 0:
            all_tokens = all_tokens[:max_size]

        # 构建 token_to_id 和 id_to_token
        self.token_to_id = {t: i for i, t in enumerate(all_tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    def __len__(self):
        return len(self.token_to_id)

    def token2id(self, token: str) -> int:
        return self.token_to_id.get(token, self.token_to_id.get("<unk>"))

    def id2token(self, id_: int) -> str:
        return self.id_to_token.get(id_, "<unk>")


class BasicTokenizer:
    def __init__(self, vocab, tokenizer=lambda text: [token.strip() for token in re.split(r'\s+|(?=[，。！？,.!?:;])|(?<=[，。！？,.!?:;])', text.strip()) if token.strip()]):
        self.vocab = vocab
        self.tokenizer = tokenizer

    def tokenize(self, text: str) -> list[int]:
        textList = self.tokenizer(text)
        ret = [self.vocab.token2id(token) for token in textList]
        return ret
    
    def detokenize(self, token_ids: list[int]) -> str:
        tokens = [self.vocab.id2token(id_) for id_ in token_ids]
        return ' '.join(tokens)

def build_counter(
        source,
        tokenizer=lambda text: [token.strip() for token in re.split(r'\s+|(?=[，。！？,.!?:;])|(?<=[，。！？,.!?:;])', text.strip()) if token.strip()],
) -> Counter:
    counter = Counter()
    print("Building token frequency counter...")
    for text in source:
        tokens = tokenizer(text)
        counter.update(tokens)
    return counter

def build_tokenizer_from_wmt_en(
        limit: int = 100000,
        max_size: int = -1,
        min_freq: int = 5,
        tokenizer=lambda text: [token.strip() for token in re.split(r'\s+|(?=[，。！？,.!?:;])|(?<=[，。！？,.!?:;])', text.strip()) if token.strip()],
        force_rebuild: bool = False
) -> BasicTokenizer:
    # 正常情况下 tokenizer 也应该加入 key 计算，但这里为了简化就不加了
    key = f"wmt_en_{limit}"
    print(f"Building/loading tokenizer with key: {key}")
    e = cacheExist(key)
    if e and not force_rebuild:
        counter = cacheLoad(key)
    else:
        counter = build_counter(source=iter_wmt_en(limit), tokenizer=tokenizer)
        cacheSave(counter, key)
    vocab = Vocab(
        counter=counter,
        max_size=max_size,
        min_freq=min_freq
    )
    tokenizer = BasicTokenizer(vocab, tokenizer=tokenizer)
    return tokenizer

def build_tokenizer_from_wmt_zh(
        limit: int = 10000,
        max_size: int = -1,
        min_freq: int = 5,
        tokenizer=lambda text: [token.strip() for token in re.split(r'\s+|(?=[，。！？,.!?:;])|(?<=[，。！？,.!?:;])', text.strip()) if token.strip()],
        force_rebuild: bool = False
) -> BasicTokenizer:
    # 正常情况下 tokenizer 也应该加入 key 计算，但这里为了简化就不加了
    key = f"wmt_zh_{limit}"
    print(f"Building/loading tokenizer with key: {key}")
    e = cacheExist(key)
    if e and not force_rebuild:
        counter = cacheLoad(key)
    else:
        counter = build_counter(source=iter_wmt_zh(limit), tokenizer=tokenizer)
        cacheSave(counter, key)
    vocab = Vocab(
        counter=counter,
        max_size=max_size,
        min_freq=min_freq
    )
    tokenizer = BasicTokenizer(vocab, tokenizer=tokenizer)
    return tokenizer

def build_counter_from_pair(
        source,
        tokenizer=lambda text: [token.strip() for token in re.split(r'\s+|(?=[，。！？,.!?:;])|(?<=[，。！？,.!?:;])', text.strip()) if token.strip()],
) -> Counter:
    counter = Counter()
    print("Building token frequency counter...")
    for text in source[0]:
        tks = tokenizer(text)
        counter.update(tks)
    for text in source[1]:
        tks = tokenizer(text)
        counter.update(tks)
    return counter

def build_tokenizer_from_wmt_en_zh(
        limit: int = 10000,
        max_size: int = -1,
        min_freq: int = 5,
        tokenizer=lambda text: [token.strip() for token in re.split(r'\s+|(?=[，。！？,.!?:;])|(?<=[，。！？,.!?:;])', text.strip()) if token.strip()],
        force_rebuild: bool = False
) -> BasicTokenizer:
    # 正常情况下 tokenizer 也应该加入 key 计算，但这里为了简化就不加了
    key = f"wmt_en_zh_{limit}"
    print(f"Building/loading tokenizer with key: {key}")
    e = cacheExist(key)
    if e and not force_rebuild:
        counter = cacheLoad(key)
    else:
        counter = build_counter_from_pair(
            source=(iter_wmt_en(limit), iter_wmt_zh(limit)),
            tokenizer=tokenizer
        )
        cacheSave(counter, key)
    vocab = Vocab(
        counter=counter,
        max_size=max_size,
        min_freq=min_freq
    )
    tokenizer = BasicTokenizer(vocab, tokenizer=tokenizer)
    return tokenizer

def test():
    limit = 1000000

    print("--- English Tokenizer Test ---")
    tokenizer = build_tokenizer_from_wmt_en(limit=limit)

    text = "hello unknown token ! . ,"
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.detokenize(tokens)

    print("len(vocab):", len(tokenizer.vocab))
    print("Text:", text)
    print("Tokens:", tokens)
    print("Token IDs:", token_ids)

    print("\n--- Chinese Tokenizer Test ---")
    tokenizer_zh = build_tokenizer_from_wmt_zh(limit=limit)

    text_zh = "你好 ，未知 的 标点符号 电脑 手机 古人 ！ 。 ,"
    tokens_zh = tokenizer_zh.tokenize(text_zh)
    token_ids_zh = tokenizer_zh.detokenize(tokens_zh)

    print("len(vocab):", len(tokenizer_zh.vocab))
    print("Text (ZH):", text_zh)
    print("Tokens (ZH):", tokens_zh)
    print("Token IDs (ZH):", token_ids_zh)

    print("\n--- English-Chinese Tokenizer Test ---")
    tokenizer_en_zh = build_tokenizer_from_wmt_en_zh(limit=limit)

    text_en_zh = ("hello unknown token ! . ,", "你好 ，未知 的 标点符号 电脑 手机 古人 ！ 。 ,")
    tokens_en_zh = tokenizer_en_zh.tokenize(text_en_zh[0]) + tokenizer_en_zh.tokenize(text_en_zh[1])
    token_ids_en_zh = tokenizer_en_zh.detokenize(tokens_en_zh)

    print("len(vocab):", len(tokenizer_en_zh.vocab))
    print("Text (EN-ZH):", text_en_zh)
    print("Tokens (EN-ZH):", tokens_en_zh)
    print("Token IDs (EN-ZH):", token_ids_en_zh)

if __name__ == "__main__":
    test()