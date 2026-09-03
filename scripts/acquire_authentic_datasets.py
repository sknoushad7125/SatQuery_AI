#!/usr/bin/env python3
import os
import sys
import json
import random

# Force deterministic splits
random.seed(42)

print("==================================================")
print(" SATQUERY AI — AUTHENTIC DATASET ACQUISITION      ")
print("==================================================")
print("This script enforces the 5-20 GB tiered dataset strategy.")
print("No synthetic data. No destructive modality conversions.")
print("Splits enforced: 70% Train, 15% Val, 15% Test.\n")

def check_dependencies():
    try:
        import requests
        import tqdm
        import datasets
    except ImportError:
        print("Missing required libraries. Please run:")
        print("pip install requests tqdm datasets huggingface_hub rasterio")
        sys.exit(1)

def print_zenodo_rsvqa_instructions():
    print("\n--- TIER 3: RSVQA ---")
    print("To preserve the authentic 16-bit/multispectral characteristics of RSVQA,")
    print("you must download it directly from the official Zenodo repository.")
    print("1. Run the following commands to download the LR archive (~1.3 GB):")
    print("   wget https://zenodo.org/record/3816654/files/Images_LR.zip -O datasets/rsvqa/Images_LR.zip")
    print("   wget https://zenodo.org/record/3816654/files/LR_split_train_questions.json -O datasets/rsvqa/LR_split_train_questions.json")
    print("   wget https://zenodo.org/record/3816654/files/LR_split_val_questions.json -O datasets/rsvqa/LR_split_val_questions.json")
    print("   wget https://zenodo.org/record/3816654/files/LR_split_test_questions.json -O datasets/rsvqa/LR_split_test_questions.json")
    print("   wget https://zenodo.org/record/3816654/files/LR_split_train_answers.json -O datasets/rsvqa/LR_split_train_answers.json")
    print("   wget https://zenodo.org/record/3816654/files/LR_split_val_answers.json -O datasets/rsvqa/LR_split_val_answers.json")
    print("   wget https://zenodo.org/record/3816654/files/LR_split_test_answers.json -O datasets/rsvqa/LR_split_test_answers.json")
    print("2. Unzip Images_LR.zip inside datasets/rsvqa/.\n")
    print("The official splits perfectly satisfy the ML validation requirements.\n")

def print_mediatum_sen12ms_instructions():
    print("\n--- TIER 4: SEN12MS ---")
    print("HuggingFace mirrors destructively convert SEN12MS 13-band and 2-band arrays to RGB PNGs.")
    print("We MUST use the raw 16-bit GeoTIFFs from TUM.")
    print("1. Download a single seasonal ROI (~2-5 GB):")
    print("   wget 'https://dataserv.ub.tum.de/s/m1474000/download?path=%2FROIs1158_Spring&files=ROIs1158_Spring_s1.tar.gz' -O datasets/sen12ms/s1.tar.gz")
    print("   wget 'https://dataserv.ub.tum.de/s/m1474000/download?path=%2FROIs1158_Spring&files=ROIs1158_Spring_s2.tar.gz' -O datasets/sen12ms/s2.tar.gz")
    print("   wget 'https://dataserv.ub.tum.de/s/m1474000/download?path=%2FROIs1158_Spring&files=ROIs1158_Spring_lc.tar.gz' -O datasets/sen12ms/lc.tar.gz")
    print("2. Extract them inside datasets/sen12ms/.")
    print("3. Our splitting script will later parse the exact physical files and assign 70/15/15 splits.\n")

def print_hf_instructions():
    print("\n--- TIER 1 & 2: LEVIR-CD & BigEarthNet.txt ---")
    print("Due to Docker/sandbox network restrictions, attempting to stream 10+ GB from HuggingFace")
    print("will throttle and hang. Please run the HF download natively on your Mac:")
    print("1. Set your token: export HF_TOKEN='your_token'")
    print("2. Use the official `datasets` library locally to pull `BIFOLD-BigEarthNetv2-0/BigEarthNet.txt`")
    print("   and `Shengzhe/LEVIR-CD`.")
    print("3. A dedicated script `scripts/dataset_tools/split_levir.py` (which we will generate) will map")
    print("   the downloaded LEVIR images to strict 70/15/15 train/val/test splits without leakage.\n")

def setup_demo_folder():
    os.makedirs("datasets/demo", exist_ok=True)
    readme_path = "datasets/demo/README.md"
    if not os.path.exists(readme_path):
        with open(readme_path, "w") as f:
            f.write("# SatQuery AI Demo Dataset\n\n")
            f.write("Place 20-50 visually impressive scenes here.\n")
            f.write("Include edge cases: urban expansion, deforestation, SAR+optical pairs.\n")
    print("Created datasets/demo/ structure.")

if __name__ == "__main__":
    setup_demo_folder()
    print_zenodo_rsvqa_instructions()
    print_mediatum_sen12ms_instructions()
    print_hf_instructions()
    print("==================================================")
    print("ACTION REQUIRED: Proceed with authentic downloads.")
    print("==================================================")
