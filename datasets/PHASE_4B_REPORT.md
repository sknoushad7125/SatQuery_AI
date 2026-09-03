# Phase 4B: LEVIR-CD Change Detection Training Calibration

## 1. Implementation
To resolve the zero-prediction collapse caused by the ~96% background imbalance, the training objective was updated from simple binary cross-entropy to a combined BCE + Dice loss.

- **Files Changed/Created**:
  - `src/losses.py` [NEW]: Implemented `DiceLoss` and `BCEDiceLoss` to handle numerical stability and unnormalized logits securely.
  - `src/train.py` [MODIFIED]: Replaced `BCEWithLogitsLoss` with `BCEDiceLoss`, increased epoch count to 5, and updated the checkpoint save path.
  - `scripts/tiny_overfit_bcedice.py` [NEW]: Verified the new loss on 2 training pairs to ensure non-zero intersection boundaries could actually be learned.
  - `scripts/independent_val_bcedice.py` [NEW]: Independent evaluation script calculating explicit precision, recall, and foreground percentage.

- **Parameters**:
  - Backbone: `timm.resnet18`
  - Loss: `BCEDiceLoss`
  - Epochs: 5
  - Batch size: 4
  - Learning rate: 1e-4

## 2. Tiny Overfit Validation
Before full training, the combined BCE + Dice loss was tested on exactly 2 samples for 50 iterations:
- **Initial Loss**: 1.8716
- **Final Loss**: 0.0488
- **Validation IoU**: 0.5916
- **Predicted Foreground**: 5.55%
*Conclusion*: The architecture rapidly learned spatial targets using the new loss formulation, clearing the way for full training.

## 3. Training & Independent Validation Results
The model was trained on all 445 LEVIR-CD training pairs and evaluated on the 64 validation pairs.

**Old Baseline vs BCE + Dice Comparison:**

| Metric               | Old BCE Baseline | BCE + Dice (5 Epochs) |
| -------------------- | ---------------: | --------------------: |
| Validation IoU       |           0.0000 |              0.4612 |
| Validation F1        |           0.0000 |              0.6312 |
| Precision            |           0.0000 |              0.5388 |
| Recall               |           0.0000 |              0.7619 |
| Predicted Positive % |          0.0000% |              5.9336% |
| Training epochs      |                1 |                     5 |
| Loss                 |              BCE |            BCE + Dice |

## 4. Status and Recommendations
**Status**: SUCCESS
The change detection pipeline has successfully escaped the zero-collapse local minimum and produces non-zero structural intersections. Performance is bounded by the 5-epoch training duration relative to standard 200-epoch literature standards, but the capability is structurally sound and explicitly verified.

**Recommendation**:
Proceed to the Phase 5 orchestration and UI assembly using the validated `change_detection`, `single_image_vqa`, and fallback implementations.
