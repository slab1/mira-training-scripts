"""Local Gradio demo for Hoodx/mira-agent-instinct.
Run: python demo.py
Requires: pip install gradio transformers torch bitsandbytes accelerate
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gradio as gr

MODEL_ID = "Hoodx/mira-agent-instinct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto"
)

def respond(message, history):
    messages = [{"role": "user", "content": message}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)
    outputs = model.generate(inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
    return tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

with gr.ChatInterface(respond, title="Mira AI Agent – Instinct Mode") as demo:
    demo.launch()
