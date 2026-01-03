# src/tokenizer/bpe.py

import tiktoken

def build_bpe_tokenizer(model_name: str = "gpt2"):
    tokenizer = tiktoken.get_encoding(model_name)
    return tokenizer

if __name__ == "__main__":
    tokenizer = build_bpe_tokenizer("gpt2")
    sample_text = "Hello, world! This is a test of the BPE tokenizer."
    token_ids = tokenizer.encode(sample_text)
    print("Token IDs:", token_ids)
    decoded_text = tokenizer.decode(token_ids)
    print("Decoded Text:", decoded_text)

    print("Vocab Size:", tokenizer.n_vocab)
    print(tokenizer.encode("<|endoftext|>"))