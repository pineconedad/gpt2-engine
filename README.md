# GPT-2 training on OpenWebText

Minimal workspace to download GPT-2, fetch OpenWebText, extract, and tokenize.

## Scripts
- `download_gpt2.py`: Cache GPT-2 model/tokenizer
- `download_corpus.py`: Download OpenWebText dataset
- `extract_corpus.py`: Extract `.tar` and decompress `.xz` into raw text
- `prepare_dataset.py`: Tokenize text into shards for training
- `gpt2_inference.py`: Quick inference sanity check

## Note
Large data and `venv/` are excluded via `.gitignore`.
