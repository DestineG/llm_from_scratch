# src/data/rawdata_handler.py

import os
import csv
from tqdm import tqdm


def wmt_zh_en_training_corpus_csv_to_txt(force_rebuild: bool = False) -> tuple[str, str]:
    """
    将 WMT 中英平行语料分别保存为 zh.txt 和 en.txt
    返回 (zh_txt_path, en_txt_path)
    """
    corpus_dir = "data/raw/wmt_zh_en_training_corpus"
    if not os.path.exists(corpus_dir):
        raise FileNotFoundError(f"{corpus_dir} does not exist")

    csv_path = os.path.join(corpus_dir, "wmt_zh_en_training_corpus.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"{csv_path} does not exist")

    zh_txt_path = os.path.join(corpus_dir, "wmt_zh_en_training_corpus.zh.txt")
    en_txt_path = os.path.join(corpus_dir, "wmt_zh_en_training_corpus.en.txt")

    def csv2txt(src, zh_dst, en_dst):
        print(f"collecting csv total lines...")
        with open(src, encoding="utf-8") as f:
            total = sum(1 for _ in f) - 1  # 减去 header

        with open(src, newline="", encoding="utf-8") as fsrc, \
             open(zh_dst, "w", encoding="utf-8") as fzh, \
             open(en_dst, "w", encoding="utf-8") as fen:

            reader = csv.reader(fsrc)
            next(reader)  # 跳过 header

            for zh, en in tqdm(reader, total=total, desc="CSV → TXT"):
                fzh.write(zh.strip() + "\n")
                fen.write(en.strip() + "\n")

    if force_rebuild or not (os.path.isfile(zh_txt_path) and os.path.isfile(en_txt_path)):
        csv2txt(csv_path, zh_txt_path, en_txt_path)

    return zh_txt_path, en_txt_path


def iter_wmt_zh_en_pairs(limit: int = -1):
    zh_path, en_path = wmt_zh_en_training_corpus_csv_to_txt()
    count = 0

    with open(zh_path, encoding="utf-8") as fzh, open(en_path, encoding="utf-8") as fen:
        while True:
            zh = fzh.readline()
            en = fen.readline()
            if not zh or not en:
                break

            yield zh.strip(), en.strip()
            count += 1

            if limit != -1 and count >= limit:
                break


def iter_wmt_zh(limit: int = -1):
    zh_path, _ = wmt_zh_en_training_corpus_csv_to_txt()
    count = 0
    with open(zh_path, encoding="utf-8") as fzh:
        for line in fzh:
            yield line.strip()
            count += 1
            if limit != -1 and count >= limit:
                break


def iter_wmt_en(limit: int = -1):
    _, en_path = wmt_zh_en_training_corpus_csv_to_txt()
    count = 0
    with open(en_path, encoding="utf-8") as fen:
        for line in fen:
            yield line.strip()
            count += 1
            if limit != -1 and count >= limit:
                break


# head data/raw/wmt_zh_en_training_corpus/wmt_zh_en_training_corpus.txt
# head data/raw/wmt_zh_en_training_corpus/wmt_zh_en_training_corpus.en.txt
# head data/raw/wmt_zh_en_training_corpus/wmt_zh_en_training_corpus.zh.txt
if __name__ == "__main__":
    print("=== 迭代前5条中英文对 ===")
    for i, (zh, en) in enumerate(iter_wmt_zh_en_pairs(limit=5)):
        print(f"{i+1}\t{zh}\t{en}")

    print("\n=== 迭代前5条中文 ===")
    for i, zh in enumerate(iter_wmt_zh(limit=5)):
        print(f"{i+1}\t{zh}")

    print("\n=== 迭代前5条英文 ===")
    for i, en in enumerate(iter_wmt_en(limit=5)):
        print(f"{i+1}\t{en}")
