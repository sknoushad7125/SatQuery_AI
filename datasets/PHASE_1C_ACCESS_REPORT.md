# Phase 1C Access and Schema Verification Report

## 1. Access Status Summary
- **BIFOLD-BigEarthNetv2-0/BigEarthNet.txt**: `ACCESS PASS` (Verified via `load_dataset` with Hugging Face token)
- **jirvin16/TEOChatlas**: `ACCESS PASS` (Verified via `HfApi.list_repo_files` with Hugging Face token)
- **xiang709/VRSBench**: `ACCESS PASS` (Public, HTTP Range extraction previously verified)

## 2. BigEarthNet.txt (BIFOLD-BigEarthNetv2-0/BigEarthNet.txt)
* **Schema Verified**: Yes (Text metadata only)
* **Actual Parquet Schema**:
  - `ID`: int64
  - `s1_name`: string (e.g. `S1B_IW_GRDH_1SDV_20170612T165809_33UUP_26_57`)
  - `patch_id`: string (e.g. `S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57`)
  - `input`: string (e.g. `Would you say that any arable land lies next to pastures in the image?`)
  - `output`: string (e.g. `yes`)
  - `type`: string (binary, multiple-choice, etc.)
  - `category`: string (adjacency, presence, etc.)
  - `split`: string (train, val, test)
  - `latitude` / `longitude`: float64
  - `country` / `season` / `climate_zone`: string
* **S1 / S2 Array Shapes & Dtypes**: NOT PRESENT. The `BigEarthNet.txt` HF repository strictly hosts the textual `.parquet` annotations. It references the Sentinel-1 and Sentinel-2 images via string IDs.
* **Deterministic Streaming Feasibility**: Streaming exactly 2,000 paired **S1/S2/text** records directly from this specific HF repository is **IMPOSSIBLE**, as it lacks the physical image payloads. Acquiring images requires downloading the raw imagery from `bigearth.net` (or another dataset host) and compiling it locally into an LMDB database using the `rico-hdl` tool as directed by their official README.
* **Required Credentials**: Local HF token (configured successfully).
* **Estimated Storage**: Annotations only (Parquet is ~200MB). Full imagery requires hundreds of GBs via an external download pipeline.

## 3. CDVQA (jirvin16/TEOChatlas)
* **Exact Files Discovered**:
  - `eval/CDVQA.json`
  - `eval/External_images.tar.gz` (contains unified test images for CDVQA, UCMerced, and others)
* **Exact Size**: 5,853.58 MB (5.85 GB) for `External_images.tar.gz`.
* **Required Credentials**: Hugging Face user agreement accepted + Local HF Token (configured successfully).
* **Contains Required Images**: Yes, this is the deduplicated source artifact used by the TEOChat-CDVQA framework to evaluate the 968 physical CDVQA test crops.
* **Estimated Storage Required**: ~5.85 GB compressed.

## 4. VRSBench
* **Existing State Check**: We previously acquired `VRSBench_EVAL_referring.json` which yields exactly 9,318 unique required validation images.
* **Location Verified**: Confirmed that all 9,318 required images exist within the `Images_val.zip` archive.
* **Estimated Storage Required**: ~4.3 GB compressed. (The 12 GB `Images_train.zip` archive is not required for our grounding evaluation).

---

## 5. Recommended Next Acquisition Order
Since BigEarthNet imagery requires external downloading from `bigearth.net` (bypassing HF native datasets) and LMDB preprocessing, it is the most time-consuming. 

**Recommended Sequence:**
1. **CDVQA Evaluation Artifact**: Download `jirvin16/TEOChatlas` -> `External_images.tar.gz` (5.8 GB) and extract only the `/CDVQA/` directory.
2. **VRSBench Evaluation Artifact**: Download `xiang709/VRSBench` -> `Images_val.zip` (4.3 GB).
3. **BigEarthNet Pipeline Setup**: Finalize whether we should use an alternative pre-compiled HF image mirror (e.g. `earthnets/BigEarthNetV2`) mapped against `BigEarthNet.txt`, or if we must proceed with the official multi-TB `bigearth.net` source and local `rico-hdl` compilation.
