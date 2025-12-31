# src/utils/cache.py

import os
import pickle

CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def cache_path(key: str) -> str:
    """返回缓存文件路径"""
    return os.path.join(CACHE_DIR, f"{key}.pkl")

def cacheExist(key: str) -> bool:
    """判断缓存文件是否存在"""
    return os.path.isfile(cache_path(key))

def cacheLoad(key: str):
    """加载缓存对象"""
    with open(cache_path(key), "rb") as f:
        obj = pickle.load(f)
    print(f"[cache] Loaded {key}")
    return obj

def cacheSave(obj, key: str):
    """保存对象到缓存"""
    with open(cache_path(key), "wb") as f:
        pickle.dump(obj, f)
    print(f"[cache] Saved {key}")
