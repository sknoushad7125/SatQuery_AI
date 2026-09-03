# REAL MODEL VALIDATION REPORT

| Model            | Dataset                | Real Data? | Train Samples | Val Samples | Metric      | Score | Checkpoint | Status |
| ---------------- | ---------------------- | ---------: | ------------: | ----------: | ----------- | ----: | ---------- | ------ |
| VQA              | RSVQA                  | No | 0 | 0 | Loss | - | - | BLOCKED |
| Change Detection | LEVIR-CD               | No | 0 | 0 | IoU         | - | - | BLOCKED |
| Change Detection | LEVIR-CD               | No | 0 | 0 | F1          | - | - | BLOCKED |
| Optical-SAR      | SEN12MS                | No | 0 | 0 | Accuracy/F1 | - | - | BLOCKED |
| Grounding        | VRSBench/base          | No | 0 | 0 | IoU         | - | - | BASELINE |

## Inference Verification
**Status Note**: Since the actual open datasets (LEVIR-CD, SEN12MS, RSVQA) are not currently mounted in `/app/datasets/`, the legitimate training scripts explicitly block execution. Synthetic Dev-Test runs are isolated behind `--synthetic-dev-mode` and their outputs are explicitly flagged as `"synthetic_dev_only"` during inference, preventing fabricated reporting.

Therefore, end-to-end execution of trained weights currently relies on the architectural fallback (Zero-shot DINOv2 / Untrained Feature Extractors / Zero-shot OWL-ViT) until the real datasets are mounted.

Competition Readiness: **NOT READY** (Waiting for Dataset Injection).
