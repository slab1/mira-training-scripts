# Kaggle Notebook for Mira Unsloth Fine-tune

Fine-tuning script for Mira (`Hoodx/mira-agent-instinct` based on `Qwen/Qwen3-8B`) using Unsloth on Kaggle T4 GPU.

## Setup Instructions
1. Open Kaggle -> **New Notebook**.
2. Set Runtime Accelerator to **GPU T4 x2** or **P100**.
3. Under **Input**, upload your dataset (`data/train.jsonl`) as a Kaggle Dataset named `mira-dataset`.

## Kaggle Training Code

```python
# 1. Install Unsloth & dependencies
!pip install -q unsloth bitsandbytes transformers accelerate datasets trl peft

import os
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 2. Configuration
MODEL_NAME = "Qwen/Qwen3-8B"
OUTPUT_DIR = "mira-finetuned"
MAX_SEQ_LENGTH = 1024

# Check dataset location
dataset_path = "/kaggle/input/mira-dataset/train.jsonl"
if not os.path.exists(dataset_path):
    dataset_path = "data/train.jsonl"  # Fallback local path

print(f"Loading dataset from: {dataset_path}")
dataset = load_dataset("json", data_files=dataset_path, split="train")

# 3. Load model in 4-bit with Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_rslora=True,
)

# 4. Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    fp16=True,
    optim="adamw_8bit",
    report_to="none",
)

# 5. Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,
    args=training_args,
)

# 6. Run Training & Save
trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved successfully to {OUTPUT_DIR}")
```
