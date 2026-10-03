# Kaggle Notebook for Mira Unsloth Fine-tune

## Setup
1. New Notebook -> GPU
2. Upload dataset to Kaggle Datasets

## Code
!pip install unsloth bitsandbytes transformers accelerate datasets trl peft

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-8B",
    max_seq_length=1024,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=8)

# Load dataset from Kaggle input
from datasets import load_dataset
dataset = load_dataset("json", data_files="/kaggle/input/mira-dataset/train.jsonl", split="train")

# Train
from trl import SFTTrainer
from transformers import TrainingArguments
training_args = TrainingArguments(
    output_dir="mira-finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
)
trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, dataset_text_field="text", max_seq_length=1024, args=training_args)
trainer.train()

# Save
model.save_pretrained("mira-finetuned")
