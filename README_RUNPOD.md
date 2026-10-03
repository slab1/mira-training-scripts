# RunPod Unsloth Training for Mira

## Prerequisites
- RunPod account
- HuggingFace token with write access to `Hoodx/mira-agent-instinct`

## 1. Create Pod
- Template: PyTorch 2.2 + CUDA 12.1
- GPU: RTX 4090 24GB or A5000 24GB
- Storage: 50 GB NVMe
- Container: `runpod/pytorch:2.2.0-py3.11-cuda12.1`

## 2. Upload files
Upload these files to pod home:
- requirements.txt
- train_unsloth_mira.py
- accelerate_config.yaml
- dataset_prep.py
- runpod_startup.sh

## 3. Prepare dataset
Create `data/raw.jsonl` with lines like:
```json
{"instruction":"Summarize the following text","input":"Long text here","output":"Short summary"}
{"instruction":"Translate to French","input":"Hello world","output":"Bonjour le monde"}
```
Upload to pod `data/raw.jsonl`

## 4. Run
```bash
chmod +x runpod_startup.sh
bash runpod_startup.sh
```

## 5. Push results
```bash
huggingface-cli login
git clone https://huggingface.co/Hoodx/mira-agent-instinct
cp -r mira-agent-instinct-finetuned/* Hoodx/mira-agent-instinct/
cd Hoodx/mira-agent-instinct
git add .
git commit -m "feat: unsloth qlora fine-tune"
git push
```

## Notes
- Training time ~ 2-6h for 3 epochs on RTX 4090 depending on dataset size
- Monitor VRAM with `nvidia-smi`
- Set `MAX_SEQ_LENGTH` in script if OOM
