# SatQuery AI — Tiered Dataset Strategy

This document outlines the strict 5–20 GB dataset architecture for the SIH26167 prototype. We strictly forbid data fabrication, synthetic arrays, and destructive RGB conversion of multispectral/SAR data.

## Strict Anti-Fabrication Rule
If any previously created dataset files are synthetic, dummy, randomly generated, RGB-converted substitutes, or otherwise non-authentic, they are identified explicitly and excluded from all validation, training, and evaluation. We do not silently count them as acquired data.

## Compute Architecture
- **Local MacBook Pro (Intel i9):** Development, API/Backend, Inference, Frontend, Validation scripts, small-scale testing.
- **Cloud GPU (e.g., Colab/Kaggle):** Heavy LoRA training, VLM adaptation, and full-scale change detection training. Trained adapters/checkpoints are brought back to the Mac for local inference.

## Tiered Datasets

### Tier 1 — Core Training: LEVIR-CD
*   **Target:** Acquire enough original image pairs to create approximately 500–1000 legitimate 256×256 patches, while preserving a meaningful number of original pairs for validation/test. Do not treat 500–1000 crops as 500–1000 independent scenes.
*   **Purpose:** Change detection, building change, temporal reasoning, CDVQA-derived questions.
*   **Splits:** 70% Train / 15% Validation / 15% Test. Splits MUST be performed on the original large image pairs *before* cropping to prevent data leakage.
*   **Storage Target:** 0.8–1.5 GB.

### Tier 2 — Multimodal Remote Sensing: BigEarthNet.txt
*   **Target:** ~2,000–5,000 genuine image-text examples. Must be independently verified from actual distribution rather than assuming HF is authoritative.
*   **Purpose:** Image-text alignment, remote-sensing captioning, semantic retrieval.
*   **Storage Target:** 2–8 GB.

### Tier 3 — VQA: RSVQA (Low Resolution)
*   **Target:** ~2,000–5,000 genuine question-answer examples.
*   **Purpose:** Demonstration of satellite image QA (e.g., "How many buildings are visible?").
*   **Storage Target:** 1–3 GB.
*   **Source:** Official Zenodo Archive (Preserving original `.tif` formats).

### Tier 4 — SAR + Optical: SEN12MS
*   **Target:** 500–2,000 genuine S1/S2 samples.
*   **Purpose:** Multimodal fusion, proving S1 SAR + S2 Optical capabilities without relying solely on RGB.
*   **Storage Target:** 1–5 GB.
*   **Source:** Official MediaTUM archives (e.g., specific ROI seasons like `ROIs1158_Spring`).

### Tier 5 — Grounding: VRSBench
*   **Target:** 100–500 genuine images/annotations.
*   **Purpose:** Evaluation and visual grounding demonstrations (e.g., bounding box overlay).
*   **Storage Target:** 0.5–2 GB.

### Demo Dataset (`datasets/demo/`)
*   **Target:** 20–50 carefully curated, visually impressive scenes.
*   **Purpose:** High-impact demonstrations (urban expansion, deforestation, SAR+optical pairs).
*   **Storage Target:** 0.2–1 GB.
