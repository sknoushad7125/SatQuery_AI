# REAL DATA TRAINING REPORT

| Component | Dataset | Real Data | Training Completed | Validation | Checkpoint Loaded | Status |
| --------- | ------- | --------- | ------------------ | ---------- | ----------------- | ------ |
| VQA | RSVQA | No | No | Not Evaluated | No (Zero-shot fallback) | BLOCKED |
| Grounding | VRSBench | No | No | Not Evaluated | No (Zero-shot fallback) | BASELINE |
| Change Detection | LEVIR-CD | No | No | Not Evaluated | No (Zero-shot fallback) | BLOCKED |
| Optical-SAR Fusion | SEN12MS | No | No | Not Evaluated | No (Untrained fallback) | BLOCKED |

**Context**: Training successfully executes and natively expects the formats specified in `DATASET_SCHEMA_VALIDATION.md`, but strictly aborts currently due to the datasets not being mounted on disk, preserving integrity and avoiding fabricated metrics.
