import sys
import os
import torch
import math

sys.path.append(os.getcwd())
from training.scripts.segmentation_metrics import SegmentationMetrics

def verify():
    metrics = SegmentationMetrics()
    
    # Target: 
    # [0, 1]
    # [2, 3]
    targets = torch.tensor([[[0, 1], [2, 3]]], dtype=torch.long)
    
    # Prediction: 
    # [0, 1]
    # [2, 2]  <- Error here, predicted 2 instead of 3
    preds = torch.tensor([[[0, 1], [2, 2]]], dtype=torch.long)
    
    # Convert preds to dummy logits [1, 4, 2, 2]
    # We put large values where the prediction is supposed to be
    logits = torch.zeros(1, 4, 2, 2)
    logits[0, 0, 0, 0] = 10.0
    logits[0, 1, 0, 1] = 10.0
    logits[0, 2, 1, 0] = 10.0
    logits[0, 2, 1, 1] = 10.0  # Predicted 2 for target 3
    
    metrics.update(logits, targets)
    res = metrics.compute()
    
    # Validation checks
    assert res['pixel_accuracy'] == 0.75, f"Expected Pixel Acc 0.75, got {res['pixel_accuracy']}"
    
    # Class 0: Int=1, Uni=1 -> IoU=1.0, Dice=1.0
    assert math.isclose(res['per_class_iou']['vegetation'], 1.0, abs_tol=1e-5)
    assert math.isclose(res['per_class_dice']['vegetation'], 1.0, abs_tol=1e-5)
    
    # Class 1: Int=1, Uni=1 -> IoU=1.0, Dice=1.0
    assert math.isclose(res['per_class_iou']['built-up area'], 1.0, abs_tol=1e-5)
    
    # Class 2: Int=1, Preds=2, Targets=1 -> Uni=2 -> IoU=0.5, Dice=0.6666...
    assert math.isclose(res['per_class_iou']['water body'], 0.5, abs_tol=1e-5)
    assert math.isclose(res['per_class_dice']['water body'], 2/3, abs_tol=1e-5)
    
    # Class 3: Int=0, Preds=0, Targets=1 -> Uni=1 -> IoU=0.0, Dice=0.0
    assert math.isclose(res['per_class_iou']['bare land'], 0.0, abs_tol=1e-5)
    assert math.isclose(res['per_class_dice']['bare land'], 0.0, abs_tol=1e-5)
    
    # Macro averages
    expected_miou = (1.0 + 1.0 + 0.5 + 0.0) / 4
    expected_mdice = (1.0 + 1.0 + 2/3 + 0.0) / 4
    assert math.isclose(res['mIoU'], expected_miou, abs_tol=1e-5)
    assert math.isclose(res['mDice'], expected_mdice, abs_tol=1e-5)
    
    print("METRICS VERIFICATION PASS")

if __name__ == '__main__':
    verify()
