import os
import sys
import glob
import json
import math
from typing import Iterator, List

import numpy as np
from tqdm import tqdm
from transformers import GPT2TokenizerFast

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACTED_DIR = os.path.join(BASE_DIR, "data", "openwebtext", "extracted")
OUT_DIR = os.path.join(BASE_DIR, "data", "tokenized")

# Config
BLOCK_SIZE = 1024  # sequence length for training
SHARD_TOKENS = BLOCK_SIZE * 1024  # ~1M tokens per shard (adjust as needed)
ENCODING = "utf-8"


def iter_text_files(root_dir: str) -> Iterator[str]:
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(".txt"):
                yield os.path.join(dirpath, fname)


def read_file(path: str) -> str:
    with open(path, "r", encoding=ENCODING, errors="ignore") as f:
        return f.read()


def save_shard(tokens: np.ndarray, shard_idx: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    bin_path = os.path.join(out_dir, f"train_{shard_idx:05d}.npy")
    meta_path = os.path.join(out_dir, f"train_{shard_idx:05d}.json")
    np.save(bin_path, tokens)
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump({"num_tokens": int(tokens.size), "block_size": BLOCK_SIZE}, mf)


def main():
    print("=" * 60)
    print("Tokenizing OpenWebText for GPT-2 training")
    print("=" * 60)

    if not os.path.isdir(EXTRACTED_DIR):
        print(f"Missing extracted dir: {EXTRACTED_DIR}")
        sys.exit(1)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.model_max_length = 100000000000  # disable warnings

    all_files = list(iter_text_files(EXTRACTED_DIR))
    if not all_files:
        print("No text files found. Run extract_corpus.py first.")
        sys.exit(1)

    print(f"Found {len(all_files)} text files")

    shard_buffer: List[int] = []
    shard_idx = 0
    total_tokens = 0

    for path in tqdm(all_files, desc="Tokenizing files"):
        text = read_file(path)
        if not text:
            continue
        # Encode with special tokens off; we train on raw token stream
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        shard_buffer.extend(ids)
        
        # Flush to shards when enough tokens accumulated
        while len(shard_buffer) >= SHARD_TOKENS:
            chunk = np.array(shard_buffer[:SHARD_TOKENS], dtype=np.int32)
            save_shard(chunk, shard_idx, OUT_DIR)
            total_tokens += chunk.size
            shard_idx += 1
            shard_buffer = shard_buffer[SHARD_TOKENS:]

    # Save remaining tokens in a final shard
    if shard_buffer:
        chunk = np.array(shard_buffer, dtype=np.int32)
        save_shard(chunk, shard_idx, OUT_DIR)
        total_tokens += chunk.size

    print("\n" + "=" * 60)
    print("Tokenization Complete!")
    print(f"Total tokens: {total_tokens}")
    print(f"Shards written: {shard_idx + 1}")
    print(f"Output dir: {OUT_DIR}")
    print("Each shard is a .npy array of token ids with a .json meta.")
    print("=" * 60)


if __name__ == "__main__":
    main()
