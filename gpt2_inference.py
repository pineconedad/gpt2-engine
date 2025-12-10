from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

# Load model and tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Example prompt
test_prompt = "Once upon a time,"
input_ids = tokenizer.encode(test_prompt, return_tensors="pt")

# Generate text (inference)
with torch.no_grad():
    output_ids = model.generate(input_ids, max_length=50, do_sample=True, top_k=50, top_p=0.95)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Prompt:", test_prompt)
print("Generated:", generated_text)
