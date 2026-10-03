"""
Prepare dataset for Mira fine-tune.
Expects input JSONL with {"instruction": "...", "input": "...", "output": "..."}
Outputs JSONL with {"text": "..."} in Qwen3 chat format.
"""
import json
from pathlib import Path

IN_PATH = Path("data/raw.jsonl")
OUT_PATH = Path("data/train.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def format_qwen3(instruction, input_text, output):
    user_content = instruction
    if input_text:
        user_content += f"\n\n{input_text}"
    return f"<|im_start|>user\n{user_content}\n<|im_end|>\n<|im_start|>assistant\n{output}\n<|im_end|>"

with open(IN_PATH, "r", encoding="utf-8") as f_in, open(OUT_PATH, "w", encoding="utf-8") as f_out:
    for line in f_in:
        obj = json.loads(line)
        text = format_qwen3(obj.get("instruction",""), obj.get("input",""), obj.get("output",""))
        f_out.write(json.dumps({"text": text}) + "\n")

print(f"Wrote {OUT_PATH}")
