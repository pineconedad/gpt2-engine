"""
GPT-2 Training Script

Optimized for RTX 4090 (24GB VRAM)
- Mixed precision (FP16) training
- Gradient accumulation for larger effective batch sizes
- Checkpointing in HuggingFace format (compatible with Magikarp tool)
- Comprehensive logging for analysis

Usage:
    python train_gpt2.py                           # Train GPT-2 Medium (default)
    python train_gpt2.py --model_size small        # Train GPT-2 Small
    python train_gpt2.py --resume checkpoints/epoch_2  # Resume from checkpoint
"""

import os
import sys
import json
import time
import math
import argparse
import random
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    get_cosine_schedule_with_warmup,
)


# ============== Configuration ==============

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data" / "tokenized"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"

# Model configurations (standard GPT-2 sizes)
MODEL_CONFIGS = {
    "small": dict(n_embd=768, n_layer=12, n_head=12),     # 124M params
    "medium": dict(n_embd=1024, n_layer=24, n_head=16),   # 355M params
    "large": dict(n_embd=1280, n_layer=36, n_head=20),    # 774M params
}

# Training defaults (optimized for RTX 4090)
DEFAULT_CONFIG = {
    "model_size": "medium",
    "block_size": 1024,           # Context length
    "batch_size": 8,              # Micro batch size (fits in VRAM)
    "gradient_accumulation": 8,   # Effective batch = 8 * 8 = 64
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "warmup_steps": 2000,
    "max_epochs": 1,
    "mixed_precision": True,
    "gradient_checkpointing": False,  # Enable if OOM
    "save_every_n_steps": 5000,   # Save checkpoint every N steps
    "log_every_n_steps": 100,     # Log metrics every N steps
    "eval_every_n_steps": 1000,   # Run evaluation every N steps
    "seed": 42,
}


# ============== Dataset ==============

class ShardedTokenDataset(Dataset):
    """
    Dataset that loads tokenized shards on-demand.
    Memory efficient for large datasets.
    """
    
    def __init__(
        self,
        data_dir: Path,
        block_size: int = 1024,
        shuffle_shards: bool = True,
        max_shards: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.block_size = block_size
        
        # Find all shards
        self.shard_paths = sorted(self.data_dir.glob("shard_*.npy"))
        if max_shards:
            self.shard_paths = self.shard_paths[:max_shards]
        
        if not self.shard_paths:
            raise ValueError(f"No shard files found in {data_dir}")
        
        if shuffle_shards:
            random.shuffle(self.shard_paths)
        
        # Calculate total tokens and samples
        self.shard_info = []
        self.total_tokens = 0
        
        print(f"Scanning {len(self.shard_paths)} shards...")
        for path in tqdm(self.shard_paths, desc="Loading shard metadata"):
            meta_path = path.with_suffix(".json")
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                    n_tokens = meta["num_tokens"]
            else:
                # Fallback: load and check
                tokens = np.load(path)
                n_tokens = len(tokens)
            
            n_samples = max(0, (n_tokens - 1) // block_size)
            self.shard_info.append({
                "path": path,
                "n_tokens": n_tokens,
                "n_samples": n_samples,
            })
            self.total_tokens += n_tokens
        
        # Build index mapping
        self.cumulative_samples = []
        total = 0
        for info in self.shard_info:
            self.cumulative_samples.append(total)
            total += info["n_samples"]
        self.total_samples = total
        
        # Cache for loaded shards
        self._current_shard_idx = -1
        self._current_tokens = None
        
        print(f"Dataset: {self.total_tokens:,} tokens, {self.total_samples:,} samples")
    
    def __len__(self):
        return self.total_samples
    
    def _load_shard(self, shard_idx: int):
        """Load a shard into memory."""
        if shard_idx != self._current_shard_idx:
            path = self.shard_info[shard_idx]["path"]
            self._current_tokens = np.load(path).astype(np.int64)
            self._current_shard_idx = shard_idx
    
    def _find_shard(self, global_idx: int) -> tuple[int, int]:
        """Find which shard contains the given global index."""
        # Binary search for the shard
        left, right = 0, len(self.cumulative_samples) - 1
        while left < right:
            mid = (left + right + 1) // 2
            if self.cumulative_samples[mid] <= global_idx:
                left = mid
            else:
                right = mid - 1
        
        shard_idx = left
        local_idx = global_idx - self.cumulative_samples[shard_idx]
        return shard_idx, local_idx
    
    def __getitem__(self, idx: int) -> dict:
        shard_idx, local_idx = self._find_shard(idx)
        self._load_shard(shard_idx)
        
        start = local_idx * self.block_size
        end = start + self.block_size + 1  # +1 for target
        
        tokens = self._current_tokens[start:end]
        
        # Handle edge case where we don't have enough tokens
        if len(tokens) < self.block_size + 1:
            tokens = np.pad(tokens, (0, self.block_size + 1 - len(tokens)))
        
        x = torch.from_numpy(tokens[:-1].copy())
        y = torch.from_numpy(tokens[1:].copy())
        
        return {"input_ids": x, "labels": y}


# ============== Training Utilities ==============

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format seconds into human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def get_lr(optimizer) -> float:
    """Get current learning rate."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


class MetricsLogger:
    """Simple CSV logger for training metrics."""
    
    def __init__(self, log_dir: Path, run_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / f"{run_name}.csv"
        self.metrics_file = self.log_dir / f"{run_name}_metrics.json"
        
        # Write header
        with open(self.log_file, "w") as f:
            f.write("step,epoch,tokens_seen,loss,learning_rate,time_per_step,gpu_memory_gb\n")
        
        self.all_metrics = []
    
    def log(self, metrics: dict):
        """Log a row of metrics."""
        self.all_metrics.append(metrics)
        
        with open(self.log_file, "a") as f:
            f.write(f"{metrics['step']},{metrics['epoch']},{metrics['tokens_seen']},"
                   f"{metrics['loss']:.6f},{metrics['learning_rate']:.2e},"
                   f"{metrics['time_per_step']:.3f},{metrics['gpu_memory_gb']:.2f}\n")
    
    def save_summary(self, summary: dict):
        """Save full metrics and summary."""
        with open(self.metrics_file, "w") as f:
            json.dump({
                "summary": summary,
                "metrics": self.all_metrics,
            }, f, indent=2)


# ============== Checkpointing ==============

def save_checkpoint(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    epoch: int,
    global_step: int,
    tokens_seen: int,
    loss: float,
    checkpoint_dir: Path,
    checkpoint_name: str,
):
    """
    Save checkpoint in HuggingFace format (compatible with Magikarp).
    """
    save_path = checkpoint_dir / checkpoint_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save model and tokenizer (HuggingFace format)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Save training state (for resuming)
    training_state = {
        "epoch": epoch,
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "loss": loss,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "timestamp": datetime.now().isoformat(),
    }
    
    torch.save(training_state, save_path / "training_state.pt")
    
    # Save human-readable metadata
    metadata = {
        "epoch": epoch,
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "tokens_seen_billions": tokens_seen / 1e9,
        "loss": loss,
        "timestamp": datetime.now().isoformat(),
        "checkpoint_name": checkpoint_name,
    }
    
    with open(save_path / "checkpoint_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  💾 Saved checkpoint: {save_path}")
    return save_path


def load_checkpoint(
    checkpoint_path: Path,
    model: GPT2LMHeadModel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    device: torch.device,
) -> dict:
    """Load checkpoint and restore training state."""
    
    # Load model weights
    model_path = checkpoint_path
    if (checkpoint_path / "model.safetensors").exists():
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path / "model.safetensors")
        # Handle weight tying: if lm_head.weight is missing, it's tied to transformer.wte.weight
        if "lm_head.weight" not in state_dict and "transformer.wte.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["transformer.wte.weight"]
        model.load_state_dict(state_dict)
    elif (checkpoint_path / "pytorch_model.bin").exists():
        state_dict = torch.load(checkpoint_path / "pytorch_model.bin", map_location=device)
        # Handle weight tying: if lm_head.weight is missing, it's tied to transformer.wte.weight
        if "lm_head.weight" not in state_dict and "transformer.wte.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["transformer.wte.weight"]
        model.load_state_dict(state_dict)
    
    # Load training state
    training_state_path = checkpoint_path / "training_state.pt"
    if training_state_path.exists():
        training_state = torch.load(training_state_path, map_location=device)
        optimizer.load_state_dict(training_state["optimizer_state_dict"])
        if scheduler and training_state.get("scheduler_state_dict"):
            scheduler.load_state_dict(training_state["scheduler_state_dict"])
        if scaler and training_state.get("scaler_state_dict"):
            scaler.load_state_dict(training_state["scaler_state_dict"])
        return training_state
    
    return {"epoch": 0, "global_step": 0, "tokens_seen": 0}


# ============== Training Loop ==============

def train(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[GradScaler],
    config: dict,
    logger: MetricsLogger,
    device: torch.device,
    start_epoch: int = 0,
    start_step: int = 0,
    start_tokens: int = 0,
):
    """Main training loop."""
    
    model.train()
    
    global_step = start_step
    tokens_seen = start_tokens
    accumulation_steps = config["gradient_accumulation"]
    block_size = config["block_size"]
    batch_size = config["batch_size"]
    tokens_per_step = block_size * batch_size * accumulation_steps
    
    # Calculate how many batches to skip when resuming
    batches_to_skip = start_step * accumulation_steps if start_step > 0 else 0
    
    best_loss = float("inf")
    
    for epoch in range(start_epoch, config["max_epochs"]):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config['max_epochs']}")
        print(f"{'='*60}")
        
        epoch_loss = 0.0
        epoch_steps = 0
        step_start_time = time.time()
        accumulated_loss = 0.0
        
        # Skip batches if resuming from checkpoint
        if batches_to_skip > 0:
            print(f"Skipping {batches_to_skip:,} batches to resume from step {start_step:,}...")
            skip_start = time.time()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Skip already-processed batches when resuming
            if batch_idx < batches_to_skip:
                if batch_idx % 100000 == 0 and batch_idx > 0:
                    elapsed = time.time() - skip_start
                    rate = batch_idx / elapsed
                    remaining = (batches_to_skip - batch_idx) / rate if rate > 0 else 0
                    print(f"  Skipped {batch_idx:,}/{batches_to_skip:,} batches ({remaining/60:.1f} min remaining)...")
                continue
            
            # Reset skip counter after first epoch
            if batch_idx == batches_to_skip and batches_to_skip > 0:
                print(f"  Skipping complete! Resuming training at batch {batch_idx:,}")
                batches_to_skip = 0  # Don't skip in subsequent epochs
            
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass with mixed precision
            if config["mixed_precision"]:
                with autocast('cuda', dtype=torch.float16):
                    outputs = model(input_ids=input_ids, labels=labels)
                    loss = outputs.loss / accumulation_steps
                
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / accumulation_steps
                loss.backward()
            
            accumulated_loss += loss.item()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                if config["mixed_precision"]:
                    scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Optimizer step
                if config["mixed_precision"]:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                if scheduler:
                    scheduler.step()
                
                optimizer.zero_grad()
                
                global_step += 1
                tokens_seen += tokens_per_step
                epoch_loss += accumulated_loss
                epoch_steps += 1
                
                # Logging
                if global_step % config["log_every_n_steps"] == 0:
                    step_time = (time.time() - step_start_time) / config["log_every_n_steps"]
                    gpu_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                    
                    logger.log({
                        "step": global_step,
                        "epoch": epoch + 1,
                        "tokens_seen": tokens_seen,
                        "loss": accumulated_loss,
                        "learning_rate": get_lr(optimizer),
                        "time_per_step": step_time,
                        "gpu_memory_gb": gpu_mem,
                    })
                    
                    progress_bar.set_postfix({
                        "loss": f"{accumulated_loss:.4f}",
                        "lr": f"{get_lr(optimizer):.2e}",
                        "tokens": f"{tokens_seen/1e9:.2f}B",
                    })
                    
                    step_start_time = time.time()
                
                # Save checkpoint at intervals
                if global_step % config["save_every_n_steps"] == 0:
                    save_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        global_step=global_step,
                        tokens_seen=tokens_seen,
                        loss=accumulated_loss,
                        checkpoint_dir=CHECKPOINT_DIR,
                        checkpoint_name=f"step_{global_step}",
                    )
                
                accumulated_loss = 0.0
        
        # End of epoch
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"\nEpoch {epoch + 1} complete. Avg Loss: {avg_epoch_loss:.4f}")
        
        # Save epoch checkpoint
        checkpoint_name = f"epoch_{epoch + 1}"
        save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch + 1,
            global_step=global_step,
            tokens_seen=tokens_seen,
            loss=avg_epoch_loss,
            checkpoint_dir=CHECKPOINT_DIR,
            checkpoint_name=checkpoint_name,
        )
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            # Also save as best model
            save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                global_step=global_step,
                tokens_seen=tokens_seen,
                loss=avg_epoch_loss,
                checkpoint_dir=CHECKPOINT_DIR,
                checkpoint_name="best",
            )
    
    return global_step, tokens_seen


# ============== Main ==============

def main():
    parser = argparse.ArgumentParser(description="Train GPT-2 on tokenized corpus")
    
    # Model
    parser.add_argument("--model_size", type=str, default=DEFAULT_CONFIG["model_size"],
                       choices=["small", "medium", "large"],
                       help="GPT-2 model size")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"],
                       help="Micro batch size")
    parser.add_argument("--gradient_accumulation", type=int, 
                       default=DEFAULT_CONFIG["gradient_accumulation"],
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG["learning_rate"],
                       help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=DEFAULT_CONFIG["max_epochs"],
                       help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=DEFAULT_CONFIG["warmup_steps"],
                       help="Warmup steps for LR scheduler")
    
    # Checkpointing
    parser.add_argument("--save_every_n_steps", type=int, 
                       default=DEFAULT_CONFIG["save_every_n_steps"],
                       help="Save checkpoint every N steps")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint path")
    
    # Performance
    parser.add_argument("--mixed_precision", action="store_true", 
                       default=DEFAULT_CONFIG["mixed_precision"],
                       help="Use mixed precision training")
    parser.add_argument("--no_mixed_precision", action="store_true",
                       help="Disable mixed precision")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       default=DEFAULT_CONFIG["gradient_checkpointing"],
                       help="Use gradient checkpointing (saves memory)")
    
    # Data
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR),
                       help="Path to tokenized data")
    parser.add_argument("--max_shards", type=int, default=None,
                       help="Limit number of shards (for testing)")
    
    # Misc
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"],
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Build config
    config = DEFAULT_CONFIG.copy()
    config.update(vars(args))
    if args.no_mixed_precision:
        config["mixed_precision"] = False
    
    # Setup
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("GPT-2 Training")
    print("=" * 60)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create model
    print(f"\nCreating GPT-2 {config['model_size']} model...")
    model_config = GPT2Config(
        vocab_size=50257,  # GPT-2 vocab size
        n_positions=config["block_size"],
        **MODEL_CONFIGS[config["model_size"]],
    )
    model = GPT2LMHeadModel(model_config)
    
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
    
    model = model.to(device)
    
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # Create dataset
    print(f"\nLoading dataset from {config['data_dir']}...")
    dataset = ShardedTokenDataset(
        data_dir=Path(config["data_dir"]),
        block_size=config["block_size"],
        shuffle_shards=True,
        max_shards=config["max_shards"],
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,  # Shards already shuffled
        num_workers=0,  # Disabled multiprocessing to avoid Windows shared memory exhaustion
        pin_memory=True,
        drop_last=True,
    )
    
    # Calculate training steps
    steps_per_epoch = len(train_loader) // config["gradient_accumulation"]
    total_steps = steps_per_epoch * config["max_epochs"]
    
    print(f"\nTraining configuration:")
    print(f"  - Model size: {config['model_size']}")
    print(f"  - Batch size: {config['batch_size']}")
    print(f"  - Gradient accumulation: {config['gradient_accumulation']}")
    print(f"  - Effective batch size: {config['batch_size'] * config['gradient_accumulation']}")
    print(f"  - Learning rate: {config['learning_rate']}")
    print(f"  - Mixed precision: {config['mixed_precision']}")
    print(f"  - Steps per epoch: {steps_per_epoch:,}")
    print(f"  - Total steps: {total_steps:,}")
    print(f"  - Tokens per step: {config['block_size'] * config['batch_size'] * config['gradient_accumulation']:,}")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.95),
    )
    
    # Create scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=total_steps,
    )
    
    # Mixed precision scaler
    scaler = GradScaler('cuda') if config["mixed_precision"] else None
    
    # Resume from checkpoint
    start_epoch = 0
    start_step = 0
    start_tokens = 0
    
    if config["resume"]:
        resume_path = Path(config["resume"])
        if resume_path.exists():
            print(f"\nResuming from checkpoint: {resume_path}")
            training_state = load_checkpoint(
                resume_path, model, optimizer, scheduler, scaler, device
            )
            start_epoch = training_state.get("epoch", 0)
            start_step = training_state.get("global_step", 0)
            start_tokens = training_state.get("tokens_seen", 0)
            print(f"  Resumed at epoch {start_epoch}, step {start_step}, tokens {start_tokens:,}")
        else:
            print(f"Warning: Checkpoint not found at {resume_path}, starting fresh")
    
    # Create logger
    run_name = f"gpt2_{config['model_size']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = MetricsLogger(LOG_DIR, run_name)
    
    # Save config
    config_path = LOG_DIR / f"{run_name}_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nLogging to: {LOG_DIR / run_name}.csv")
    print(f"Checkpoints: {CHECKPOINT_DIR}")
    
    # Train!
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        final_step, final_tokens = train(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            logger=logger,
            device=device,
            start_epoch=start_epoch,
            start_step=start_step,
            start_tokens=start_tokens,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted! Saving checkpoint...")
        save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=start_epoch,
            global_step=start_step,
            tokens_seen=start_tokens,
            loss=0.0,
            checkpoint_dir=CHECKPOINT_DIR,
            checkpoint_name="interrupted",
        )
        sys.exit(0)
    
    total_time = time.time() - start_time
    
    # Save summary
    logger.save_summary({
        "total_steps": final_step,
        "total_tokens": final_tokens,
        "total_time_seconds": total_time,
        "model_size": config["model_size"],
        "n_parameters": n_params,
    })
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"  Total steps: {final_step:,}")
    print(f"  Total tokens: {final_tokens:,} ({final_tokens/1e9:.2f}B)")
    print(f"  Total time: {format_time(total_time)}")
    print(f"  Checkpoints saved to: {CHECKPOINT_DIR}")
    print(f"  Logs saved to: {LOG_DIR}")


if __name__ == "__main__":
    main()
