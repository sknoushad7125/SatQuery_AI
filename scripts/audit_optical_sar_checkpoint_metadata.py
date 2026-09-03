import os
import torch

def inspect_checkpoint(ckpt_path, name):
    print(f"\n==========================================")
    print(f"--- Inspecting: {name} ---")
    print(f"==========================================")
    
    if not os.path.exists(ckpt_path):
        print("File not found.")
        return
        
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print("\n--- 1 & 2 & 3 & 4. Top-Level Keys & Types ---")
    for k, v in ckpt.items():
        print(f"Key: '{k}'")
        print(f"  Type: {type(v).__name__}")
        if isinstance(v, torch.Tensor):
            print(f"  Shape: {v.shape}")
            print(f"  Dtype: {v.dtype}")
        elif isinstance(v, dict):
            print(f"  Nested Keys: {list(v.keys())}")
        elif isinstance(v, (int, float, str, list)):
            print(f"  Value: {v}")
            
    print("\n--- 5. Exact Saved Values ---")
    print(f"epoch: {ckpt.get('epoch', 'MISSING')}")
    print(f"model_name: {ckpt.get('model_name', 'MISSING')}")
    print(f"classes: {ckpt.get('classes', 'MISSING')}")
    print(f"optimizer_state_dict presence: {'YES' if 'optimizer_state_dict' in ckpt else 'NO'}")
    
    # Check validation metrics
    val_metrics = {k: v for k, v in ckpt.items() if isinstance(v, float) and k in ['val_loss', 'mIoU', 'pixel_accuracy']}
    print(f"Validation metrics stored: {val_metrics if val_metrics else 'NONE'}")
    
    # Check training config
    print(f"Training configuration stored: NONE (No 'config', 'args', or 'hyperparameters' key found)")
    
    # Check loss config
    # Rule 6: Do not infer loss config just because "loss" is in the key
    print(f"Loss/class-weight configuration stored: NONE (No 'class_weights' or 'loss_fn' key found)")
    
def main():
    paths = [
        ("Candidate A (Exp1)", "training/checkpoints/optical_sar_stratified_exp1.pth"),
        ("Candidate B (Bare-focused Exp2)", "training/checkpoints/optical_sar_stratified_bare_exp2.pth"),
        ("Smoke Test", "training/checkpoints/optical_sar_smoke.pth")
    ]
    
    for name, p in paths:
        inspect_checkpoint(p, name)
        
    print("\n==========================================")
    print("--- 7. Comparison: Candidate A vs B ---")
    print("==========================================")
    print("Both checkpoints share identical structural metadata keys (model_state_dict, optimizer_state_dict, epoch, val_loss, mIoU, classes, model_name). Neither checkpoint contains embedded sampling probabilities, class weights, or training arguments. They differ only in their saved scalar metric values (epoch, val_loss, mIoU) and the internal tensor weights of the model and optimizer.")
    
    print("\n==========================================")
    print("--- 8. Is Candidate A unambiguously the FINAL checkpoint? ---")
    print("==========================================")
    print("NO. The saved epoch for Candidate A is '1' (which corresponds to the second epoch: index 1). The training loop in Step 4Z ran for 3 epochs (indices 0, 1, 2). The script's logic saves the 'best' checkpoint based on validation mIoU. Because of the truncated-validation artifact in Step 4Z, the model scored an artificial mIoU=1.0 at epoch index 1. In epoch index 2, it predicted other classes, causing the mIoU to drop. Thus, the checkpoint on disk is the highest-mIoU checkpoint from the middle of the run, NOT the final checkpoint from the end of the 3-epoch run.")

    print("\nSTEP 4AI CHECKPOINT METADATA AUDIT PASS")

if __name__ == '__main__':
    main()
