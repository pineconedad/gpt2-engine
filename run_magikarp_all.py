"""
Run Magikarp analysis on all checkpoints to track undertrained tokens over training.
"""

import os
import subprocess
import sys
from pathlib import Path

# Configuration
CHECKPOINTS_DIR = "checkpoints"
MAGIKARP_SCRIPT = "magikarp-main/magikarp/fishing.py"
KNOWN_UNUSED_TOKENS = "[177,178,179,180,181,182,183,184,185,186,187]"

def get_step_checkpoints():
    """Get all step_* checkpoints sorted by step number."""
    checkpoints = []
    for item in os.listdir(CHECKPOINTS_DIR):
        if item.startswith("step_") and os.path.isdir(os.path.join(CHECKPOINTS_DIR, item)):
            try:
                step_num = int(item.replace("step_", ""))
                checkpoints.append((step_num, item))
            except ValueError:
                continue
    
    # Sort by step number
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

def run_magikarp(checkpoint_name):
    """Run Magikarp on a single checkpoint."""
    checkpoint_path = f"{CHECKPOINTS_DIR}/{checkpoint_name}"
    
    cmd = [
        sys.executable,
        MAGIKARP_SCRIPT,
        "--model_id", checkpoint_path,
        "--known_unused_tokens", KNOWN_UNUSED_TOKENS
    ]
    
    print(f"\n{'='*60}")
    print(f"Running Magikarp on: {checkpoint_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0 or result.returncode == 1  # 1 is OK (LaTeX error)
    except Exception as e:
        print(f"Error running Magikarp on {checkpoint_name}: {e}")
        return False

def main():
    checkpoints = get_step_checkpoints()
    
    print(f"Found {len(checkpoints)} step checkpoints:")
    for step_num, name in checkpoints:
        tokens_trained = step_num * 64 * 1024  # effective_batch * seq_len
        print(f"  - {name} (~{tokens_trained/1e9:.2f}B tokens)")
    
    print(f"\nStarting Magikarp analysis on all checkpoints...")
    print(f"Results will be saved to: results/")
    
    successful = 0
    failed = 0
    
    for step_num, checkpoint_name in checkpoints:
        success = run_magikarp(checkpoint_name)
        if success:
            successful += 1
        else:
            failed += 1
            print(f"WARNING: Failed on {checkpoint_name}")
    
    print(f"\n{'='*60}")
    print(f"COMPLETED")
    print(f"{'='*60}")
    print(f"Successful: {successful}/{len(checkpoints)}")
    print(f"Failed: {failed}/{len(checkpoints)}")
    print(f"\nResults saved to: results/")

if __name__ == "__main__":
    main()
