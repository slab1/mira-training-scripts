# Training Mira AI Agent Model - Options 1-3

## Option 1: RunPod - Recommended
Cost: ~$0.60/hr RTX 4090
Steps:
1. Create pod PyTorch 2.2 CUDA 12.1
2. Upload files from /tmp/mira-training
3. bash runpod_startup.sh

## Option 2: Lambda Labs
Cost: ~$0.90/hr RTX 6000 Ada
Steps:
1. Create instance
2. SSH in
3. pip install -r requirements.txt
4. python train_unsloth_mira.py

## Option 3: HuggingFace AutoTrain
Cost: Free credits then pay
Steps:
1. Go to huggingface.co/autotrain
2. Upload dataset
3. Select Qwen/Qwen3-8B base
4. Choose LoRA
5. Train and push to Hoodx/mira-agent-instinct
