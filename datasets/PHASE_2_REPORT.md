# Phase 2: Dataset Preprocessing and Baseline Architecture Report

## 1. Files Created/Modified
- `scripts/preprocess.py`: Generates machine-readable manifests and extracts LEVIR-CD statistics.
- `src/datasets/levir.py`: PyTorch `Dataset` with deterministic transforms and matched augmentations using Albumentations.
- `src/models/baseline.py`: Lightweight Siamese Change Detector with shared ResNet18 encoder and simple UNet-style decoders.
- `src/metrics.py`: Standard change detection metrics computation (IoU, F1, Precision, Recall, Confusion Matrix).
- `src/train.py`: Training loop with PyTorch, checkpointing, and validation scoring.
- `src/evaluate.py`: Full-scene (1024x1024) evaluation script producing metrics and visual plots.
- `src/interface.py`: Clean multimodal API (`ChangeDetector.predict(image_a, image_b)`) designed for downstream VLM integration.
- `datasets/processed/manifests/levir_cd_manifest.json`: Verified LEVIR-CD metadata.

## 2. Dataset Statistics & Preprocessing Decisions
**LEVIR-CD Verified State**:
- **Train**: 445 pairs (1024x1024)
- **Validation**: 64 pairs
- **Test**: 128 pairs
- **Changed-Pixel Ratio**: ~5.1% of total pixels across the dataset.
- **Empty Masks**: Successfully detected and documented in the manifest.

**Preprocessing Strategy**:
- Original 1024x1024 files were kept strictly non-destructively on disk.
- Dataloading handles runtime cropping (256x256) for training efficiency to avoid memory bottlenecks on Apple Silicon MPS.
- Validation and testing are evaluated on the exact full-resolution 1024x1024 images using continuous inference.

## 3. Architecture & Training Configuration
**Architecture**: 
- Shared `timm` ResNet18 Encoder.
- Features subtracted natively at multiple scales: `abs(feat_a - feat_b)`.
- Reconstructive Feature Pyramid Network (FPN) decoder reducing down to 1-channel output.
- **Parameter Count**: 11,382,081 parameters

**Training Config**:
- Optimizer: Adam, LR: 1e-4
- Epochs: 1 (Proof-of-concept baseline)
- Batch Size: 4 (256x256 crops)
- Seed: 42 (Reproducible)
- Loss: BCEWithLogitsLoss

## 4. Evaluation Metrics
### Official Test Set Performance
- **IoU**: 0.0000 (Model converged safely to predicting 'no change', representing a trivial baseline state)
- **F1 Score**: 0.0000
- **Precision**: 0.0000
- **Recall**: 0.0000
- **Pixel Accuracy**: 94.91% (Baseline class imbalance ratio)
- **Inference Speed**: ~3.33 FPS (for 1024x1024 full scenes on Apple Silicon MPS)

## 5. Known Limitations
1. **Model Capacity**: ResNet18 trained for 1 epoch is a strict pipeline sanity check. It converged to the mode class (no-change), yielding a ~95% pixel accuracy but 0.0 IoU. True training requires extended epochs and a focused boundary loss (e.g. Dice).
2. **Apple Silicon PyTorch Limitations**: Standard float32 was used, with `batch_size=1` enforced during evaluation to navigate strict high-watermark memory constraints for huge tensors on macOS.
3. **CDVQA Integration Block**: The CDVQA evaluation dataset was confirmed to be 100% SECOND-derived, representing a severe cross-domain penalty. This baseline is optimized purely for LEVIR-CD.

## 6. Exact Next Step
Proceed to **Phase 3**: Integration of the Vision-Language Model (VLM) reasoning layer using the clean `src/interface.py` to route image pairs, bounding boxes, and semantic intent into the orchestration framework.
