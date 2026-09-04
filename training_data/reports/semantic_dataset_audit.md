# SatQuery AI Semantic Dataset Audit

## 1. BigEarthNet.txt Grounding

The verified coordinate convention for BigEarthNet.txt relies on normalized bounds representing corners `[min1 min2, max1 max2]`. Our parser accurately captures these 4 bounding values. However, without official source documentation defining whether this refers to `[ymin, xmin, ymax, xmax]` or `[xmin, ymin, xmax, ymax]`, our parser blindly maps them to the required `[x1, y1, x2, y2]` output schema. This risks a silent axis swap. Thus, the bounding box extraction needs correction or external validation of the axis order before model training can consume these labels.

## 2. CDVQA Temporal Ordering

The verified temporal convention is that `*_1.png` is the BEFORE image and `*_2.png` is the AFTER image. This was empirically proven by mapping directional QA queries (e.g., "did buildings increase/decrease?") to structural edge density analyses of the image pairs. Our parser (`prepare_cdvqa.py`) correctly maps `_1` to `before_image` and `_2` to `after_image`. No correction is required for CDVQA.

## 3. Current Preprocessing Status

- Structural validation passed.
- Approximately 7.42M records were cleanly compiled without pipeline errors.
- Structural validation does NOT by itself prove semantic correctness (e.g. axis swaps, inverted temporal order).
- These two semantic checks were required before training subsets are generated to guarantee the Vision-Language Model does not learn inverted coordinates or reversed time.

## 4. Required Corrections

- **BigEarthNet Parser**: We must definitively establish whether BigEarthNet uses `ymin, xmin` or `xmin, ymin` before generating grounding training subsets. The current parser assumes the order in the text matches the output schema.

## Authoritative Verification
- **Source inspected**: Local python datamodules, READMEs, Hugging Face dataset card, and the official paper (`arXiv:2603.29630`).
- **Exact semantic definition**: None found.
- **Evidence**: The implementation strictly treats the bounding boxes as strings (`output = sample.output`) to feed directly into LLaVA-style VLMs. There is no spatial transformation logic that reveals the axes.
- **Empirical cross-check**: Impossible locally because BigEarthNet source imagery is massive and intentionally excluded from local disk (`requires_external_imagery: true`).
- **Conclusion**: The axis order is genuinely undocumented.
- **prepare_bigearthnet.py**: No modifications are made because we cannot blindly guess a conversion.

BIGEARTHNET_BBOX: UNRESOLVED
CDVQA_ORDER: VERIFIED
TRAINING_READY: NO
