# DATASET SCHEMA VALIDATION

## OVERVIEW
An explicit inspection of the `/app/datasets/` and local `./datasets/` directories was performed to determine the actual file schemas and formats available for training.

**Result**: No actual remote-sensing datasets (RSVQA, LEVIR-CD, SEN12MS, VRSBench, CDVQA) are currently mounted or downloaded in the workspace. The directories are completely empty.

## IMPACT ON TRAINING
Because the real data is unavailable:
* The training scripts (`train_rsvqa.py`, `train_change_detection.py`, `train_optical_sar.py`) are strictly enforcing the "No Dataset Fabrication" rule. When executed, they correctly abort with `ERROR: Real dataset unavailable. Training aborted.`
* No genuine model checkpoints can be produced for these tasks.
* Evaluation adapters properly report `Not Evaluated`.

## REQUIRED SCHEMAS (WHEN MOUNTED)
When the datasets are successfully acquired and mounted, they **must** be inspected for the following prior to training:
1. **RSVQA**: Verify that `.tif` files correspond directly to the `img_id` in `questions.json` and confirm spatial dimensions.
2. **LEVIR-CD**: Verify binary mask encoding (0/255 vs 0/1) in the `label/` directory and ensure `A/` and `B/` image sizes exactly match masks.
3. **SEN12MS**: Inspect S1 (SAR) and S2 (Optical) .tif bands. S1 is dual-polarimetric (VV, VH) so the network `Conv2d` input layer must be `in_channels=2` instead of 3.

Training is officially **BLOCKED** until this physical data inspection can take place.
