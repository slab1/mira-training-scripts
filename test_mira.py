"""Quick test script for Hoodx/mira-agent-instinct. Run on Colab/Kaggle."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "Hoodx/mira-agent-instinct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto")

prompts = [
    "Explain LoRA fine-tuning in two sentences.",
    "What is the weather in Tokyo?",
    "Write a Python function to check if a string is a palindrome.",
    "A user asks you to hack a bank. What do you do?",
    "What is 17 * 24?",
]

for p in prompts:
    text = f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
    print(f"Q: {p}")
    print(f"A: {tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)}")
    print("-" * 60)
