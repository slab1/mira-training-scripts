#!/usr/bin/env bash
set -e

echo "=== Mira Unsloth Training Startup ==="

# System packages
apt-get update -qq && apt-get install -y -qq git curl > /dev/null

# Python deps
pip install --upgrade pip
pip install -r requirements.txt

# Clone repo if needed
if [ ! -d "mira-agent-instinct" ]; then
  git clone https://huggingface.co/Hoodx/mira-agent-instinct.git
fi

# Prepare dataset placeholder
mkdir -p data
if [ ! -f data/raw.jsonl ]; then
  echo '{"instruction":"Hello","input":"","output":"Hi!"}' > data/raw.jsonl
  echo "Created dummy data/raw.jsonl - replace with your dataset"
fi

python dataset_prep.py

# Train
export MIRA_DATASET=data/train.jsonl
accelerate launch train_unsloth_mira.py

echo "Training complete. Artifacts in ./mira-agent-instinct-finetuned"
