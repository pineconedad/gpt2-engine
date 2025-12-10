# GPT-2 Training on OpenWebText

Complete pipeline to download, prepare, and train GPT-2 on OpenWebText corpus.

## Pipeline Scripts

| Script | Description |
|--------|-------------|
| `download_gpt2.py` | Cache GPT-2 model/tokenizer from HuggingFace |
| `download_corpus.py` | Download OpenWebText dataset (~12GB) |
| `extract_corpus.py` | Extract `.tar` and decompress `.xz` into raw text |
| `tokenize_corpus.py` | Batch tokenize text into shards for training |
| `train_gpt2.py` | Train GPT-2 (Small/Medium/Large) with checkpointing |
| `gpt2_inference.py` | Quick inference sanity check |

## Training Usage

```bash
# Train GPT-2 Medium (default, recommended for 13.5B tokens)
python train_gpt2.py

# Train GPT-2 Small (faster, for testing)
python train_gpt2.py --model_size small

# Train with custom settings
python train_gpt2.py --model_size medium --max_epochs 2 --learning_rate 1e-4

# Resume from checkpoint
python train_gpt2.py --resume checkpoints/epoch_1
```

## Checkpoints

Checkpoints are saved in HuggingFace format at:
- `checkpoints/epoch_N/` - End of each epoch
- `checkpoints/step_N/` - Every 5000 steps
- `checkpoints/best/` - Best validation loss

Each checkpoint contains model weights, tokenizer, and training state.

## Analysis with Magikarp

After training, analyze under-trained tokens:

```bash
cd magikarp-main
python -m magikarp.fishing --model_id "../checkpoints/epoch_1"
```

## Note
Large data and `venv/` are excluded via `.gitignore`.
