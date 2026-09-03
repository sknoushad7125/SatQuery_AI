import subprocess
import torch
import sys
import os

def verify():
    ckpt_path = "training/checkpoints/optical_sar_smoke.pth"
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        
    print("1. Running train_optical_sar.py smoke test...")
    cmd = [
        sys.executable, "training/scripts/train_optical_sar.py",
        "--epochs", "1",
        "--batch-size", "2",
        "--max-train-batches", "5",
        "--max-val-batches", "2",
        "--checkpoint", ckpt_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("--- Training Output ---")
    print(result.stdout)
    if result.stderr:
        print("--- Training Errors ---")
        print(result.stderr)
        
    assert result.returncode == 0, "Training script failed!"
    assert "Saved best checkpoint" in result.stdout, "Checkpoint saving not logged!"
    assert "Val Loss:" in result.stdout, "Validation not logged!"
    
    print("\n2. Verifying Checkpoint...")
    assert os.path.exists(ckpt_path), f"Checkpoint file {ckpt_path} not found!"
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    expected_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch', 'val_loss', 'classes', 'model_name']
    
    for key in expected_keys:
        assert key in ckpt, f"Checkpoint missing required key: {key}"
        
    assert ckpt['model_name'] == 'DualUNet-FeatureFusion', "Wrong model name"
    assert ckpt['classes'] == ["vegetation", "built-up area", "water body", "bare land"], "Wrong classes"
    assert "opt_encoder.conv1.weight" in ckpt['model_state_dict'], "Missing model weights"
    
    print("Checkpoint verified successfully.")
    print("\nREAL TRAINING SMOKE TEST PASS")

if __name__ == '__main__':
    verify()
