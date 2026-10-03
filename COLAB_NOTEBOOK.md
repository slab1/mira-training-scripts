# Colab Notebook for Mira Unsloth Fine-tune

## Cell 1
!pip install -q unsloth bitsandbytes transformers accelerate datasets trl peft

## Cell 2
from google.colab import files
uploaded = files.upload()
# Upload data/raw.jsonl

## Cell 3
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-8B",
    max_seq_length=1024,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=8)

## Cell 4
# Load dataset and train
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

dataset = load_dataset("json", data_files="data/train.jsonl", split="train")

training_args = TrainingArguments(
    output_dir="mira-finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    fp16=True,
)

trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset, dataset_text_field="text", max_seq_length=1024, args=training_args)
trainer.train()

## Cell 5
model.save_pretrained("mira-finetuned")
# Push to HF
from huggingface_hub import login
login()
# push code
