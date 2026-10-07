#!/usr/bin/env python3
import json, random, urllib.request
from pathlib import Path

ORIG_URL = "https://raw.githubusercontent.com/slab1/mira-training-scripts/main/data/train.jsonl"
ORIG_PATH = Path("data/original_train.jsonl")
HERMES_PATH = Path("data/hermes_train.jsonl")
OUT_PATH = Path("data/train_merged.jsonl")
AXOLOTL_OUT = Path("data/train_axolotl_merged.jsonl")

# Download original if missing
if not ORIG_PATH.exists():
    print("Downloading original train.jsonl...")
    urllib.request.urlretrieve(ORIG_URL, ORIG_PATH)

def load_jsonl(p):
    with p.open() as f:
        return [json.loads(l) for l in f if l.strip()]

original = load_jsonl(ORIG_PATH)
print(f"Original examples: {len(original)}")

if HERMES_PATH.exists():
    hermes = load_jsonl(HERMES_PATH)
    print(f"Hermes examples: {len(hermes)}")
else:
    print("Hermes file not found, using original only")
    hermes = []

merged = original + hermes
random.shuffle(merged)
with OUT_PATH.open("w") as f:
    for ex in merged:
        f.write(json.dumps(ex) + "\n")
print(f"Merged -> {OUT_PATH} : {len(merged)} examples")

# Convert to Axolotl messages format
import re
pattern = re.compile(r"<\|im_start\|>(user|assistant)\n(.*?)(?=<\|im_end\|>)", re.DOTALL)

def to_messages(obj):
    text = obj.get("text","")
    msgs = []
    for m in pattern.finditer(text):
        role = m.group(1)
        content = m.group(2).strip()
        msgs.append({"role": role, "content": content})
    if msgs and msgs[0]["role"]=="user":
        return {"messages": msgs}
    return {"text": text}

with OUT_PATH.open() as fin, AXOLOTL_OUT.open("w") as fout:
    for line in fin:
        obj = json.loads(line)
        fout.write(json.dumps(to_messages(obj)) + "\n")
print(f"Axolotl format -> {AXOLOTL_OUT}")
