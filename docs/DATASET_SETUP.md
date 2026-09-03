# Remote Sensing Datasets Setup

This project structurally supports training deep learning models on genuine remote-sensing datasets. Synthetic random data is used for codebase validation *only* if explicitly enabled.

To achieve real remote-sensing ML performance (and mark the SIH26167 modules as `COMPLETE`), follow these steps to download and mount the legitimate datasets.

## 1. LEVIR-CD (Change Detection)
**Source**: [LEVIR-CD Dataset GitHub](https://github.com/justcheneng/LEVIR-CD)
**Format**: High-resolution Google Earth images, 256x256 tiles, binary change masks.
**Directory Structure Required**:
```
/datasets/levir_cd/
    train/
        A/       # Pre-event optical
        B/       # Post-event optical
        label/   # Binary mask
    val/
        A/
        B/
        label/
```
**Training Command**:
```bash
python training/scripts/train_change_detection.py \
  --dataset_path /app/datasets/levir_cd \
  --output_dir /app/training/checkpoints/siamunet_cd \
  --batch_size 4 --epochs 20
```

## 2. SEN12MS (Optical-SAR Fusion)
**Source**: [SEN12MS Technical University of Munich](https://mediatum.ub.tum.de/1474000)
**Format**: Sentinel-1 SAR, Sentinel-2 Optical, and MODIS Land Cover.
**Directory Structure Required**:
```
/datasets/sen12ms/
    train/
        s1/  # SAR
        s2/  # Optical
        lc/  # Land cover
    val/
        s1/
        s2/
        lc/
```
**Training Command**:
```bash
python training/scripts/train_optical_sar.py \
  --dataset_path /app/datasets/sen12ms \
  --output_dir /app/training/checkpoints/optical_sar_fusion \
  --epochs 15
```

## 3. RSVQA
**Source**: [RSVQA Project](https://rsvqa.sylvainlobry.com/)
**Format**: TIF imagery and JSON QA pairs.
**Directory Structure Required**:
```
/datasets/rsvqa/
    images/
        1.tif
        2.tif
    train_questions.json
    val_questions.json
```
**Training Command**:
```bash
python training/scripts/train_rsvqa.py \
  --dataset_path /app/datasets/rsvqa \
  --output_dir /app/training/checkpoints/rsvqa_lora \
  --epochs 10
```
