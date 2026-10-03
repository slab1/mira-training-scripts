# Colab Free training for Mira
# Run in Colab with T4 GPU
!pip install -q unsloth bitsandbytes transformers accelerate datasets trl peft

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-8B",
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16)
print("Model loaded")
