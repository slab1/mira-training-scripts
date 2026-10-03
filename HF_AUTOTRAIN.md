# HuggingFace AutoTrain for Mira

## Steps
1. Go to https://huggingface.co/autotrain
2. Login with HF token
3. New Project -> Text -> Fine-tune
4. Base model: Qwen/Qwen3-8B
5. Upload dataset: data/train.jsonl with {"text": "..."}
6. Parameters:
   - Method: LoRA
   - r: 16
   - alpha: 16
   - epochs: 3
   - learning_rate: 2e-4
   - max_seq_length: 1024
7. Train
8. Push to Hoodx/mira-agent-instinct

## Dataset format
JSONL with {"text": "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>"}

## Notes
- Free credits available
- AutoTrain handles GPU allocation
- Results auto-push to Hub
