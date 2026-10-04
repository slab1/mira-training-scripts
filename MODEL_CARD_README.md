---
pipeline_tag: text-generation
library_name: transformers
language:
  - en
license: apache-2.0
tags:
  - agent
  - smolagents
  - recursive-self-improvement
  - instinct
  - AGI
  - autonomous-agents
  - qwen3
  - qlora
  - unsloth
  - rsi
base_model: Qwen/Qwen3-8B
datasets:
  - HuggingFaceH4/ultrachat_200k
---

# Mira AI Agent Model – Instinctive AGI / RSI

![Banner](assets/banner.svg)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Hoodx%2Fmira--agent--instinct-blue)](https://huggingface.co/Hoodx/mira-agent-instinct)
[![Demo](https://img.shields.io/badge/%F0%9F%8C%90%20Demo-Hoodx%2Fmira--agent--instinct--demo-green)](https://huggingface.co/spaces/Hoodx/mira-agent-instinct-demo)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

Mira is an agent-oriented LLM with instinctive reasoning compression and safe recursive self-improvement loops. Designed for long-horizon tasks, tool use, and self-play fine-tuning with eval gating.

> **✅ Weights available.** This repo now contains merged 16-bit weights (`model-0000{1..4}-of-00004.safetensors`) fine-tuned from `Qwen/Qwen3-8B` via Unsloth QLoRA. It loads directly with `AutoModelForCausalLM`. Training scripts live in [`slab1/mira-training-scripts`](https://github.com/slab1/mira-training-scripts).

## Overview

* **Instinct mode**: fast heuristic policy that reduces reasoning tokens while preserving first-attempt accuracy.
* **Agent layer**: smolagents / Transformers Agent Toolkit compatible.
* **RSI**: offline self-play fine-tuning with DPO, frozen improvement operator, lineage tracking.
* **Safety**: gated repo, guardrails, audit log, explicit limitations.

## Model Details

* Base model: Qwen/Qwen3-8B
* Architecture: decoder-only transformer
* Context length: 32k
* Training: supervised fine-tune on agent trajectories + self-play synthetic data
* Quantized variants: Q4_K_M, Q8_0

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Hoodx/mira-agent-instinct"

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Hoodx/mira-agent-instinct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
```

The LoRA adapter (`adapter_model.safetensors`, r=8, alpha=8) is also kept in the repo for reference.
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

messages = [
    {"role": "user", "content": "Plan a 3-step research workflow and execute step 1"}
]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(**inputs)
```

### smolagents

```python
from smolagents import CodeAgent, HfApiModel

model = HfApiModel(model_id="Hoodx/mira-agent-instinct")
agent = CodeAgent(model=model, tools=[...])
agent.run("...")
```

## Demo

See the HuggingFace Space: `Hoodx/mira-agent-instinct-demo`

## Evaluation

Results in `.eval_results/`:
* Terminal-Bench 2.1
* SWE-Bench Verified
* AgentBench

## Evaluation

| Metric | Value | Notes |
|--------|-------|-------|
| Perplexity (held-out) | TBD | Run on 10% held-out split of `Hoodx/mira-dataset` |
| Task completion rate | TBD | 50 agent prompts, human-rated |
| Training loss (final) | TBD | From `checkpoint-6/trainer_state.json` |

Eval script: `eval_mira.py` in `slab1/mira-training-scripts` (run on Colab/Kaggle).

## Limitations

* Not AGI. Agent-oriented LLM with heuristic compression.
* Recursive self-improvement is offline and eval-gated.
* Tool use requires guardrails.

## Citation

```bibtex
@misc{mira-agent-instinct,
  title={Mira AI Agent Model – Instinctive AGI / RSI},
  author={Hoodx},
  year={2026}
}
```

## License

Apache-2.0
