# Phase 1C Consolidated Acquisition Report

## Current Disk Space Summary
* Available Disk Space: ~570 GB
* Total Phase 1C Data Acquired: ~10 GB (TEOChatlas + VRSBench caches & extracted files)

## 1. CDVQA Physical Images (Phase 1C-A)
* **Status**: `PASS`
* **Artifact**: Downloaded `eval/External_images.tar.gz` (5.85 GB) from `jirvin16/TEOChatlas`.
* **Execution**: Extracted exclusively the CDVQA test subset (1,936 files). Manifest generated.

## 2. CDVQA ↔ LEVIR-CD Mapping (Phase 1C-B)
* **Status**: `PASS` (With Critical Scientific Finding)
* **Total unique test images**: 968
* **LEVIR-CD-derived count**: 0
* **SECOND-derived count**: 968
* **Finding**: Extensive pixel-level matching confirms that the official CDVQA Test Split contains **zero** LEVIR-CD images. It is 100% SECOND-derived. See `datasets/cdvqa/CDVQA_MAPPING_REPORT.md` for details. This means evaluating a LEVIR-CD-trained model on this test set is a zero-shot cross-domain evaluation.

## 3. VRSBench (Phase 1C-C)
* **Status**: `PASS`
* **Artifact**: Downloaded `Images_val.zip` (4.3 GB) from `xiang709/VRSBench`.
* **Execution**: Successfully ignored the 12GB training zip. Extracted the exactly required 9,318 images needed for the `VRSBench_EVAL_referring.json`. Validated 1:1 image-to-annotation mapping. See `datasets/vrsbench/VRSBENCH_IMAGE_ACQUISITION_REPORT.md`.

## 4. BigEarthNet.txt (Phase 1C-D)
* **Status**: `DEFERRED / BLOCKED`
* **Artifact**: `BIFOLD-BigEarthNetv2-0/BigEarthNet.txt`
* **Execution**: Verified that the repository holds only text pointers, not physical imagery. Drafted `datasets/bigearthnet/BIGEARTHNET_ACQUISITION_PLAN.md` documenting that acquiring imagery requires multi-terabyte raw downloads + LMDB compilation. Recommended deferring BigEarthNet to Phase 3.

## Phase Conclusion
Phase 1C is structurally complete. All locally feasible datasets for prototype evaluation (LEVIR-CD, RSVQA-LR, VRSBench, CDVQA Test) are securely on disk with validated JSON annotations. We are ready to proceed to model implementation, though we must structurally accommodate the zero-shot CDVQA discovery.
