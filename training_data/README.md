# SatQuery AI - Dataset Preprocessing & Manifests

This directory contains the unified training and validation manifests required for multi-task Vision-Language model training on the SIH26167 problem statement.

## 1. Directory Structure

- `manifests/`: Unified JSONL format files mapping the original dataset files to training-ready records.
- `reports/`: Audit logs, validation output, and statistical reports.

## 2. Dataset Specifics

### RSVQA
- **Source**: `datasets/rsvqa/`
- **Details**: 772 low-resolution (LR) images.
- **Handling**: Inactive metadata is filtered out. Since the official test/val metadata arrays contain only inactive data, we create a reproducible `internal_val` split (~10%) partitioned by image ID (seed=42) to prevent train/val leakage.

### VRSBench
- **Source**: `datasets/vrsbench/`
- **Details**: ~30K images split across training and validation zip archives.
- **Handling**: Extracted annotations mapping VQA, Captioning, and Grounding tasks. Original referring expressions and normalized bounds are preserved in `[0,1]`. Images are not unzipped and rely on archive references for storage efficiency.

### BigEarthNet
- **Source**: `datasets/bigearthnet_txt/`
- **Details**: Text annotations from BigEarthNet.txt parquet file mapped to tasks (`binary`/`mcq` -> VQA, `bounding box` -> Grounding, `captioning` -> Captioning).
- **Important limitation**: The physical raw imagery for BigEarthNet (~110GB) is NOT downloaded locally. The manifest entries explicitly declare `"image_available": false` and `"requires_external_imagery": true`. Cloud/Colab compute storage will be required to materialize these images later.

### CDVQA
- **Source**: `datasets/cdvqa/`
- **Details**: Temporal change VQA.
- **Handling**: Identifies the explicit before (`_1.png`) and after (`_2.png`) image pairs to ensure spatial reasoning mapping across image states.

## 3. Rebuilding Manifests

To cleanly rebuild all manifests and validation reports without model training:

```bash
# Execute orchestration and validation scripts
python scripts/datasets/build_manifests.py
```

This will run all underlying scripts in `scripts/datasets/`, parse datasets, generate unified JSONL manifests, and perform integrity checks reporting to `reports/validation_report.json`.
