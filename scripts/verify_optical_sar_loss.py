import sys
import os
import torch

sys.path.append(os.getcwd())
from training.scripts.train_optical_sar import SEN12MSSegmentationDataset
from training.scripts.segmentation_loss import OpticalSARLoss

def main():
    print("1. Loading training statistics from Phase 5 Step 4K...")
    # Exact pixel counts from Train split only
    counts = torch.tensor([1566218418, 248386131, 59011224, 1303651], dtype=torch.float32)
    total = counts.sum()
    frequencies = counts / total
    
    print(f"Frequencies: Veg={frequencies[0]:.4f}, Urban={frequencies[1]:.4f}, Water={frequencies[2]:.4f}, Bare={frequencies[3]:.4f}")
    
    # We use Inverse Square Root Frequency Smoothing. 
    # Standard 1/f yields an extreme weight of ~1200 for Bare, which destabilizes gradients.
    # Inverse Square Root smoothly compresses the dynamic range of weights while still strongly prioritizing rare classes.
    inv_sqrt_freq = 1.0 / torch.sqrt(frequencies)
    
    # Normalize so Vegetation (majority class) has weight = 1.0
    weights = inv_sqrt_freq / inv_sqrt_freq[0]
    
    print(f"Calculated Stable Weights: Veg={weights[0]:.4f}, Urban={weights[1]:.4f}, Water={weights[2]:.4f}, Bare={weights[3]:.4f}")
    
    print("\n2. Instantiating Loss...")
    criterion = OpticalSARLoss(class_weights=weights)
    
    print("\n3. Generating dummy data and real masks...")
    base_dir = os.path.join(os.getcwd(), 'datasets/sen12ms')
    dataset = SEN12MSSegmentationDataset(base_dir=base_dir, split='train')
    
    _, _, mask1 = dataset[0]
    _, _, mask2 = dataset[1]
    targets = torch.stack([mask1, mask2], dim=0) # [2, 256, 256]
    
    # Create dummy logits [2, 4, 256, 256] requiring grad
    logits = torch.randn(2, 4, 256, 256, requires_grad=True)
    
    print("\n4. Calculating loss...")
    loss = criterion(logits, targets)
    print(f"Loss value: {loss.item():.4f}")
    assert torch.isfinite(loss), "FAIL: Loss is not finite!"
    
    print("\n5. Running backward()...")
    loss.backward()
    
    print("Verifying gradients...")
    assert logits.grad is not None, "FAIL: Gradients are None!"
    assert torch.isfinite(logits.grad).all(), "FAIL: Gradients are not finite!"
    assert (logits.grad != 0).any(), "FAIL: Gradients are all zero!"
    
    print("Gradients are finite and non-zero. PASS.")
    
    print("\nSEGMENTATION LOSS PASS")

if __name__ == '__main__':
    main()
