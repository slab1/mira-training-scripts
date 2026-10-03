"""
Unsloth QLoRA fine-tune for Hoodx/mira-agent-instinct
Base: Qwen/Qwen3-8B
Target: Hoodx/mira-agent-instinct
"""
import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

MODEL_NAME = "Qwen/Qwen3-8B"
OUTPUT_DIR = "./mira-agent-instinct-finetuned"
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 16

# Load model in 4-bit
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    bias="none",
    use_rslora=True,
)

# Example dataset - replace with your own
# For now use a placeholder JSONL with {"text": "..."}
DATASET_PATH = os.getenv("MIRA_DATASET", "data/train.jsonl")

if os.path.exists(DATASET_PATH):
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
else:
    # Fallback dummy
    from datasets import Dataset
    dataset = Dataset.from_dict({"text": ["<|im_start|>user\nHello\n<|im_end|>\n<|im_start|>assistant\nHi!\n<|im_end|>"]})

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="adamw_8bit",
    remove_unused_columns=False,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,
    args=training_args,
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved to {OUTPUT_DIR}")
