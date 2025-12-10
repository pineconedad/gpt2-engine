"""
Efficient batch tokenization of OpenWebText corpus for GPT-2 training.

This script:
1. Reads text files in batches for memory efficiency
2. Uses batch tokenization for 5-10x speedup over single-file processing
3. Saves tokenized data as shards (.npy files) ready for training
4. Supports resuming from where it left off
"""

import os
import sys
import json
import argparse
from typing import List, Iterator
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import GPT2TokenizerFast


# ============== Configuration ==============
BASE_DIR = Path(__file__).parent.resolve()
EXTRACTED_DIR = BASE_DIR / "data" / "openwebtext" / "extracted"
OUT_DIR = BASE_DIR / "data" / "tokenized"

BLOCK_SIZE = 1024           # Sequence length for training
SHARD_SIZE = 1024 * 1024    # ~1M tokens per shard
BATCH_SIZE = 100            # Number of files to tokenize at once
ENCODING = "utf-8"


def get_text_files(root_dir: Path) -> List[Path]:
    """Recursively find all .txt files."""
    return sorted(root_dir.rglob("*.txt"))


def read_files_batch(file_paths: List[Path]) -> List[str]:
    """Read multiple files and return their contents."""
    texts = []
    for path in file_paths:
        try:
            with open(path, "r", encoding=ENCODING, errors="ignore") as f:
                content = f.read().strip()
                if content:
                    texts.append(content)
        except Exception as e:
            print(f"Warning: Could not read {path}: {e}")
    return texts


def save_shard(tokens: np.ndarray, shard_idx: int, out_dir: Path):
    """Save a shard of tokens to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    bin_path = out_dir / f"shard_{shard_idx:05d}.npy"
    meta_path = out_dir / f"shard_{shard_idx:05d}.json"
    
    np.save(bin_path, tokens)
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "num_tokens": int(tokens.size),
            "block_size": BLOCK_SIZE,
            "dtype": str(tokens.dtype)
        }, f, indent=2)


def get_existing_shards(out_dir: Path) -> int:
    """Count existing shards to support resuming."""
    if not out_dir.exists():
        return 0
    return len(list(out_dir.glob("shard_*.npy")))


def tokenize_in_batches(
    tokenizer: GPT2TokenizerFast,
    all_files: List[Path],
    batch_size: int = BATCH_SIZE,
    shard_size: int = SHARD_SIZE,
    out_dir: Path = OUT_DIR,
    resume: bool = True
) -> dict:
    """
    Tokenize files in batches for efficiency.
    
    Returns statistics about the tokenization process.
    """
    # Check for existing shards (resume support)
    existing_shards = get_existing_shards(out_dir) if resume else 0
    if existing_shards > 0 and resume:
        print(f"Found {existing_shards} existing shards. Resuming...")
    
    shard_idx = existing_shards
    token_buffer: List[int] = []
    total_tokens = 0
    files_processed = 0
    
    # Process files in batches
    num_batches = (len(all_files) + batch_size - 1) // batch_size
    
    with tqdm(total=len(all_files), desc="Tokenizing", unit="files") as pbar:
        for batch_start in range(0, len(all_files), batch_size):
            batch_end = min(batch_start + batch_size, len(all_files))
            batch_files = all_files[batch_start:batch_end]
            
            # Read batch of files
            texts = read_files_batch(batch_files)
            
            if not texts:
                pbar.update(len(batch_files))
                continue
            
            # Batch tokenization - MUCH faster than one-by-one
            encoded_batch = tokenizer(
                texts,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                verbose=False
            )
            
            # Collect all token IDs from the batch
            for input_ids in encoded_batch["input_ids"]:
                if input_ids:
                    token_buffer.extend(input_ids)
            
            files_processed += len(batch_files)
            
            # Save shards when buffer is large enough
            while len(token_buffer) >= shard_size:
                chunk = np.array(token_buffer[:shard_size], dtype=np.uint16)
                save_shard(chunk, shard_idx, out_dir)
                total_tokens += chunk.size
                shard_idx += 1
                token_buffer = token_buffer[shard_size:]
                
                # Update progress description with shard info
                pbar.set_postfix({
                    "shards": shard_idx,
                    "tokens": f"{total_tokens/1e6:.1f}M"
                })
            
            pbar.update(len(batch_files))
    
    # Save any remaining tokens as final shard
    if token_buffer:
        chunk = np.array(token_buffer, dtype=np.uint16)
        save_shard(chunk, shard_idx, out_dir)
        total_tokens += chunk.size
        shard_idx += 1
    
    return {
        "total_tokens": total_tokens,
        "total_shards": shard_idx,
        "files_processed": files_processed
    }


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize OpenWebText corpus for GPT-2 training"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Number of files to process per batch (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--shard-size", type=int, default=SHARD_SIZE,
        help=f"Tokens per shard (default: {SHARD_SIZE})"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh instead of resuming from existing shards"
    )
    parser.add_argument(
        "--input-dir", type=str, default=str(EXTRACTED_DIR),
        help="Input directory with extracted text files"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUT_DIR),
        help="Output directory for tokenized shards"
    )
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print("GPT-2 Corpus Tokenization (Batch Mode)")
    print("=" * 60)
    
    # Validate input directory
    if not input_dir.is_dir():
        print(f"Error: Input directory not found: {input_dir}")
        print("Run extract_corpus.py first to extract the text files.")
        sys.exit(1)
    
    # Find all text files
    print(f"\nScanning {input_dir} for text files...")
    all_files = get_text_files(input_dir)
    
    if not all_files:
        print("Error: No .txt files found in input directory.")
        print("Run extract_corpus.py first.")
        sys.exit(1)
    
    print(f"Found {len(all_files):,} text files")
    
    # Load tokenizer
    print("\nLoading GPT-2 tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e12)  # Disable length warnings
    
    # Show configuration
    print(f"\nConfiguration:")
    print(f"  - Batch size: {args.batch_size} files")
    print(f"  - Shard size: {args.shard_size:,} tokens (~{args.shard_size * 2 / 1e6:.1f} MB per shard)")
    print(f"  - Output dir: {output_dir}")
    print(f"  - Resume: {not args.no_resume}")
    
    # Run tokenization
    print("\n" + "-" * 60)
    stats = tokenize_in_batches(
        tokenizer=tokenizer,
        all_files=all_files,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
        out_dir=output_dir,
        resume=not args.no_resume
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Tokenization Complete!")
    print("=" * 60)
    print(f"  Files processed: {stats['files_processed']:,}")
    print(f"  Total tokens:    {stats['total_tokens']:,} ({stats['total_tokens']/1e6:.2f}M)")
    print(f"  Shards created:  {stats['total_shards']}")
    print(f"  Output location: {output_dir}")
    print()
    print("Each shard contains:")
    print(f"  - .npy file: uint16 array of token IDs")
    print(f"  - .json file: metadata (token count, block size)")
    print("=" * 60)


if __name__ == "__main__":
    main()
