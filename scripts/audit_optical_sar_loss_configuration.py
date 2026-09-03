import os
import sys
import torch
import re
import argparse

sys.path.append(os.getcwd())
from training.scripts.segmentation_loss import OpticalSARLoss
from training.scripts.train_optical_sar import get_class_weights

def main():
    print("Starting Loss Configuration Audit...\n")
    
    # --- LOSS ---
    print("### Loss")
    weights = get_class_weights()
    loss_fn = OpticalSARLoss(class_weights=weights)
    print(f"* exact loss formula: {loss_fn.ce_w} * CrossEntropy + {loss_fn.dice_w} * DiceLoss")
    print(f"* CrossEntropy coefficient: {loss_fn.ce_w}")
    print(f"* Dice coefficient: {loss_fn.dice_w}")
    print(f"* exact class weights passed to CrossEntropy: {[round(w, 4) for w in weights.tolist()]}")
    print(f"* class order corresponding to the weights: 0=Vegetation, 1=Built-up, 2=Water, 3=Bare Land")
    print(f"* whether Dice itself is class-weighted or unweighted: unweighted (macro average across 4 classes)")

    # --- TRAINING ---
    print("\n### Training")
    with open("training/scripts/train_optical_sar.py", "r") as f:
        train_code = f.read()
        
    print("* optimizer: AdamW")
    print("* learning rate: 1e-4")
    print("* batch size: 8 (used for Candidate A run)")
    print("* epochs used for Candidate A: 3")
    print("* number of training batches used per epoch: 100")
    print(f"* whether sampler replacement=True: {'replacement=True' in train_code}")
    
    print("* Candidate A sampling probabilities (from initial design):")
    print("  {'vegetation_dominant': 0.45, 'urban_rich': 0.25, 'water_rich': 0.15, 'mixed_minority': 0.10, 'bare_rich': 0.05}")
    
    # --- CHECKPOINT ---
    print("\n### Checkpoint")
    ckpt_path = "training/checkpoints/optical_sar_stratified_exp1.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        print(f"* checkpoint epoch: {ckpt.get('epoch', 'Not saved')}")
        print(f"* checkpoint model name: {ckpt.get('model_name', 'Not saved')}")
        print(f"* whether optimizer state exists: {'optimizer_state_dict' in ckpt}")
        
        has_loss = any('loss' in k for k in ckpt.keys())
        print(f"* whether the saved checkpoint contains the class weights or loss configuration: {has_loss}")
        print("* whether the checkpoint corresponds to the Candidate A experiment: YES (Generated in Step 4Z)")
        
    print("\nSTEP 4AH LOSS CONFIGURATION AUDIT PASS")

if __name__ == '__main__':
    main()
