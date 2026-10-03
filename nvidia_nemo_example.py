"""
Nvidia NeMo fine-tuning example for Mira AI agent model
Base: Qwen/Qwen3-8B
Requires: Nvidia GPU with CUDA 12.1+, NeMo installed
"""
import nemo.collections.nlp as nemo_nlp
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager

# Example config for NeMo LLM fine-tuning with LoRA
config = {
    "model": {
        "type": "Qwen3ForCausalLM",
        "pretrained_model_name": "Qwen/Qwen3-8B",
        "lora_r": 16,
        "lora_alpha": 16,
    },
    "trainer": {
        "devices": 1,
        "accelerator": "gpu",
        "precision": "bf16-mixed",
    },
    "data": {
        "train_dataset": "data/train.jsonl",
        "format": "jsonl",
        "text_field": "text",
    },
    "optim": {
        "lr": 2e-4,
        "epochs": 3,
    }
}

# Load model
model = nemo_nlp.models.LLM.from_pretrained(
    model_name="Qwen/Qwen3-8B",
    lora_r=16,
    lora_alpha=16,
)

# Fine-tune
model.train(
    train_dataset="data/train.jsonl",
    max_seq_length=1024,
    batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_epochs=3,
)

# Save
model.save_pretrained("mira-nemo-finetuned")
print("Saved NeMo fine-tuned model to mira-nemo-finetuned")
