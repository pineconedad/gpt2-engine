from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Download and cache GPT-2 model weights and tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

print("Model and tokenizer loaded!")
