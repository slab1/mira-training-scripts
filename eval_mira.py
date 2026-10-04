"""Eval script for Hoodx/mira-agent-instinct. Run on Colab/Kaggle."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import math

MODEL_ID = "Hoodx/mira-agent-instinct"
DATASET = "Hoodx/mira-dataset"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto")
model.eval()

ds = load_dataset(DATASET, split="train")
# hold out last 10%
split = ds.train_test_split(test_size=0.1, seed=42)
eval_ds = split["test"]

total_loss = 0
count = 0
for ex in eval_ds:
    inputs = tokenizer(ex["text"], return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])
    total_loss += out.loss.item()
    count += 1

avg_loss = total_loss / count
perplexity = math.exp(avg_loss)
print(f"Eval examples: {count}")
print(f"Avg loss: {avg_loss:.4f}")
print(f"Perplexity: {perplexity:.2f}")
