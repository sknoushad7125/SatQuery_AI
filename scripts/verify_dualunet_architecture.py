import sys
import os
import torch

sys.path.append(os.getcwd())
from backend.tools.optical_sar.real_fusion import OpticalSARFusionNet

def verify():
    print("1. Instantiating DualUNet model...")
    model = OpticalSARFusionNet(num_classes=4)
    
    # Enable gradient tracking on inputs to verify backprop
    print("\n2. Creating dummy tensors...")
    opt_x = torch.randn(2, 13, 256, 256, requires_grad=True)
    sar_x = torch.randn(2, 2, 256, 256, requires_grad=True)
    
    print(f"Optical input shape: {opt_x.shape}")
    print(f"SAR input shape: {sar_x.shape}")
    
    print("\n3. Running forward pass...")
    logits = model(opt_x, sar_x)
    
    print("\n4. Verifying output shape...")
    expected_shape = (2, 4, 256, 256)
    assert logits.shape == expected_shape, f"FAIL: Expected {expected_shape}, got {logits.shape}"
    print(f"Output shape is exactly {logits.shape}. PASS.")
    
    print("\n5. Parameter count:")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    print("\n6. Running backward pass to verify gradients...")
    # Dummy loss
    loss = logits.sum()
    loss.backward()
    
    # Check gradients
    if opt_x.grad is None or (opt_x.grad == 0).all():
        print("FAIL: No gradients flowing to Optical (S2) input!")
        sys.exit(1)
    else:
        print("Optical (S2) gradient verified. PASS.")
        
    if sar_x.grad is None or (sar_x.grad == 0).all():
        print("FAIL: No gradients flowing to SAR (S1) input!")
        sys.exit(1)
    else:
        print("SAR (S1) gradient verified. PASS.")
        
    print("\nDUALUNET ARCHITECTURE PASS")

if __name__ == '__main__':
    verify()
