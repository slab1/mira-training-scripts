# Setup Guides for Mira Training

## Nvidia NeMo
```bash
pip install -r requirements_nemo.txt
# Or use NGC container
docker run --gpus all -it nvcr.io/nvidia/nemo/llm:24.10
python nvidia_nemo_example.py
```

## Nvidia AI Workbench
```bash
# Install AI Workbench
curl -fsSL https://raw.githubusercontent.com/NVIDIA/ai-workbench/main/install.sh | bash
# Open Workbench UI, create project, import repo
# Use NGC PyTorch container
```

## Nvidia NIM
```bash
pip install nvidia-nim
nimctl login
nimctl deploy qwen3-8b
```

## Nvidia Base Command
```bash
pip install base-command
bc login
bc project create mira-training
bc job submit --config train_config.yaml
```

## NGC Containers
```bash
docker pull nvcr.io/nvidia/pytorch:24.10-py3
docker run --gpus all -v $(pwd):/workspace nvcr.io/nvidia/pytorch:24.10-py3
pip install -r requirements.txt
python train_unsloth_mira.py
```
