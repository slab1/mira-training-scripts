"""
Unsloth QLoRA fine-tune for Hoodx/mira-agent-instinct
Base: Qwen/Qwen3-8B
Target: Hoodx/mira-agent-instinct
"""
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Train Mira AI model with Unsloth QLoRA")
    parser.add_argument("--dataset", type=str, default=os.getenv("MIRA_DATASET", "data/train.jsonl"), help="Path to training dataset JSONL")
    parser.add_argument("--output_dir", type=str, default="./mira-agent-instinct-finetuned", help="Output directory for fine-tuned model")
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset and configuration without running full GPU training")
    args = parser.parse_args()

    print(f"Dataset path: {args.dataset}")
    print(f"Output directory: {args.output_dir}")

    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file {args.dataset} not found.")
        sys.exit(1)

    if args.dry_run:
        import json
        with open(args.dataset, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"Dry run check: {len(lines)} examples found in {args.dataset}.")
        for i, line in enumerate(lines[:3]):
            data = json.loads(line)
            assert "text" in data, f"Line {i} missing 'text' field"
        print("Dry run validation successful.")
        return

    import torch
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    MODEL_NAME = "Qwen/Qwen3-8B"
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

    dataset = load_dataset("json", data_files=args.dataset, split="train")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
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
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")

if __name__ == "__main__":
    main()
