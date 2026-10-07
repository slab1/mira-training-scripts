#!/usr/bin/env python3
import json, re, sys
from pathlib import Path

INPUT = Path("data/original_train.jsonl")
OUTPUT = Path("data/train_axolotl.jsonl")

# Regex to split Qwen chat markers
pattern = re.compile(r"<\|im_start\|>(user|assistant)\n(.*?)(?=<\|im_end\|>)", re.DOTALL)

def parse_text(text):
    msgs = []
    for m in pattern.finditer(text):
        role = m.group(1)
        content = m.group(2).strip()
        msgs.append({"role": role, "content": content})
    # Ensure alternating user/assistant and start with user
    if not msgs or msgs[0]["role"] != "user":
        return None
    return msgs

count = 0
with INPUT.open() as fin, OUTPUT.open("w") as fout:
    for line in fin:
        obj = json.loads(line)
        text = obj.get("text","")
        msgs = parse_text(text)
        if msgs and len(msgs) >= 2:
            fout.write(json.dumps({"messages": msgs}) + "\n")
            count += 1
        else:
            # fallback: keep as raw text
            fout.write(json.dumps({"text": text}) + "\n")
print(f"Converted {count} examples to messages format -> {OUTPUT}")
