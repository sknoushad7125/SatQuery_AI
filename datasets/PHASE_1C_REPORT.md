# Phase 1C Dataset Acquisition Report

## 1. Files created/modified
- `scripts/acquire_bigearthnet.py` (Created, halted on auth check)
- `scripts/verify_bigearthnet.py` (Created, halted on auth check)
- `scripts/acquire_vrsbench.py` (Created, implements `fsspec` selective ZIP extraction)
- `scripts/verify_vrsbench.py` (Created, parses schema and checks manifest)
- `datasets/vrsbench/qa/VRSBench_EVAL_referring.json` (Acquired)
- `datasets/vrsbench/qa/VRSBench_train.json` (Acquired)

## 2. Commands executed
- **HF Token Verification**:
  - `test -n "$HF_TOKEN" && echo "HF_TOKEN_PRESENT" || echo "HF_TOKEN_MISSING"`
  - `cat ~/.cache/huggingface/token`
  - Python `huggingface_hub.whoami()` (Failed: `LocalTokenNotFoundError`)
- **VRSBench Investigation**:
  - `curl` fetched the VRSBench remote directory via HF API.
  - Downloaded `VRSBench_EVAL_referring.json`.
  - Ran `jq` on `VRSBench_EVAL_referring.json` to extract `obj_corner` and `ground_truth`.
  - Python scripts using `fsspec` and `zipfile` HTTP Range Requests to read `Images_val.zip` central directory without downloading the payload.
  - Python script to cross-reference unique image IDs from annotations against the `Images_val` remote ZIP manifest.

## 3. BigEarthNet
* **Authentication status**: MISSING (LocalTokenNotFoundError)
* **Source revision**: N/A (Blocked)
* **Actual schema**: N/A (Blocked)
* **Actual S1 shape/dtype**: N/A (Blocked)
* **Actual S2 shape/dtype**: N/A (Blocked)
* **Actual text schema**: N/A (Blocked)
* **Selected records**: N/A (Blocked)
* **Train/val/test counts**: N/A (Blocked)
* **Disk usage**: 0
* **Validation results**: BLOCKED

## 4. VRSBench
* **Annotation source**: `xiang709/VRSBench` (EVAL JSONs directly fetched)
* **Actual split counts**: 16,159 EVAL referring expressions.
* **Unique image count**: 9,350 total in `Images_val.zip`.
* **Required image count for our evaluation**: 9,318 exactly (all confirmed to exist exclusively within `Images_val.zip`). No images from the 12GB `Images_train.zip` are required for evaluation.
* **Bounding-box schema**: 
  - Bin format `ground_truth`: `{<xmin><ymin><xmax><ymax>}` discretized 0-100.
  - Coordinate format `obj_corner`: 8-point float representation `[x1, y1, x2, y2, x3, y3, x4, y4]` normalized 0.0-1.0.
* **Archive structure**: Central flat directory structure (e.g., `Images_val/P0003_0002.png`).
* **Whether selective extraction is technically possible**: **YES**. Python `fsspec` + `zipfile` successfully reads the central directory via HTTP Range Requests and extracts targeted images perfectly.
* **Estimated download/storage requirement**: ~4.3 GB. Since we need 9,318 out of the 9,350 images in `Images_val.zip` (99.6%), performing 9,318 distinct HTTP range requests is extremely inefficient compared to a single bulk download of the 4.3 GB archive. The download was deferred as requested.

## 5. Updated Project Status
- **LEVIR-CD**: `COMPLETE`
- **RSVQA-LR**: `COMPLETE`
- **CDVQA**: `PARTIAL` (Blocked by massive WebDataset redundancy; requires HF auth for deduplicated TEOChatlas artifact)
- **BigEarthNet**: `BLOCKED` (Missing Hugging Face Token)
- **VRSBench**: `PARTIAL` (Annotations acquired, schema verified, large image download deferred)

## 6. Recommendation
Phase 1C is **incomplete** due to missing Hugging Face authentication, which strictly blocks BigEarthNet acquisition and the CDVQA TEOChatlas artifact. 

**Next Action**: Do not proceed to model implementation yet. The user must configure their Hugging Face token locally (via `huggingface-cli login` or environment variables) and authorize access to `BIFOLD-BigEarthNetv2-0` and `jirvin16/TEOChatlas`. Once authenticated, we should resume Phase 1C to acquire BigEarthNet, the CDVQA test subset, and the 4.3 GB VRSBench `Images_val.zip`.
