# Mira AI Training Scripts

Training scripts for fine-tuning Mira AI agent model `Hoodx/mira-agent-instinct` with Qwen/Qwen3-8B using Unsloth QLoRA.

## Quick Start Options

### 1. Google Colab Free
- Notebook: `mira_colab.ipynb`
- GPU: T4 15GB
- Steps: Upload notebook, change runtime to T4, upload dataset

### 2. Kaggle Notebooks
- Guide: `KAGGLE_NOTEBOOK.md`
- GPU: T4/P100, 30 hrs/week free

### 3. HuggingFace AutoTrain
- Guide: `HF_AUTOTRAIN.md`
- Free credits, managed GPU

### 4. RunPod / Lambda Labs
- Script: `runpod_startup.sh`
- Cost: ~$0.60/hr RTX 4090

## Nvidia Open Source Training Options

Nvidia provides open source tools for training AI agent models:

- **NeMo** - https://github.com/NVIDIA/NeMo
  Open source toolkit for building conversational AI, LLMs, and agents. Supports fine-tuning with NeMo Guardrails.

- **NIM Microservices** - https://github.com/NVIDIA/nim
  Deploy and fine-tune models with Nvidia NIM.

- **AI Workbench** - https://github.com/NVIDIA/ai-workbench
  Open source AI development environment for training and deploying models on Nvidia GPUs.

- **Base Command** - https://github.com/NVIDIA/base-command
  MLOps platform for training at scale.

- **NGC Containers** - https://ngc.nvidia.com
  Pre-built containers for PyTorch, Transformers, NeMo.

To train Mira with Nvidia stack:
1. Use NeMo for fine-tuning Qwen3 with LoRA
2. Use AI Workbench for local GPU training
3. Use NGC PyTorch container with Unsloth

## Files
- `train_unsloth_mira.py` - Main training script
- `requirements.txt` - Dependencies
- `dataset_prep.py` - Dataset formatting
- `accelerate_config.yaml` - Accelerate config
- `mira_colab.ipynb` - Colab notebook
