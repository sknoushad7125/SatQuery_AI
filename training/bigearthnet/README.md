# BigEarthNet Adaptation Pipeline

This directory contains reproducible scripts for fine-tuning a vision-language model (VLM) using the BigEarthNet dataset.

## Setup
```bash
pip install -r ../../requirements.txt
```

## Steps
1. **Prepare Data**: Run `python prepare.py` to download/format the dataset.
2. **Train LoRA**: Run `python train_lora.py` to perform parameter-efficient fine-tuning on a base VLM (e.g., LLaVA or a smaller alternative).
3. **Evaluate**: Run `python evaluate.py`

*Note: These scripts are designed to be run on a machine with sufficient GPU memory (e.g., 24GB+ VRAM) and will save checkpoints to the `checkpoints/` directory.*
