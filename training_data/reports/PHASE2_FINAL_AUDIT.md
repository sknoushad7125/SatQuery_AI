# Phase 2 Final Audit

## Exact subset counts
- `vqa_train.jsonl`: 137333
- `vqa_val.jsonl`: 43112
- `caption_train.jsonl`: 20264
- `caption_val.jsonl`: 9350
- `grounding_train.jsonl`: 25590
- `grounding_val.jsonl`: 16159
- `change_vqa_train.jsonl`: 30000
- `change_vqa_val.jsonl`: 4058

## Dataset composition
| subset | dataset | records |
|---|---|---|
| caption_train.jsonl | vrsbench | 20264 |
| caption_val.jsonl | vrsbench | 9350 |
| change_vqa_train.jsonl | cdvqa | 30000 |
| change_vqa_val.jsonl | cdvqa | 4058 |
| grounding_train.jsonl | vrsbench | 25590 |
| grounding_val.jsonl | vrsbench | 16159 |
| vqa_train.jsonl | rsvqa | 51520 |
| vqa_train.jsonl | vrsbench | 85813 |
| vqa_val.jsonl | rsvqa | 5703 |
| vqa_val.jsonl | vrsbench | 37409 |

*Note: BigEarthNet grounding count is strictly 0.*

## Missing fields
- VQA missing questions: 0
- VQA missing answers: 0
- Caption missing captions: 0
- Grounding missing text/query: 2 (in `grounding_train.jsonl`)
- Grounding missing bbox: 0
- Change VQA missing before image: 0
- Change VQA missing after image: 0
- Change VQA missing question: 0
- Change VQA missing answer: 0

## Duplicate records
- Duplicate `sample_id` (all subsets): 0
- Duplicate image references (within same subset):
  - `caption_train.jsonl`: 2
  - `caption_val.jsonl`: 0
  - `change_vqa_train.jsonl`: 58258
  - `change_vqa_val.jsonl`: 7922
  - `grounding_train.jsonl`: 9891
  - `grounding_val.jsonl`: 6841
  - `vqa_train.jsonl`: 116558
  - `vqa_val.jsonl`: 33706
- Duplicate `before_image + after_image` (Change VQA only):
  - `change_vqa_train.jsonl`: 29129
  - `change_vqa_val.jsonl`: 3961

*(Note: Duplicate images and pairs are expected semantics for dense VQA datasets where multiple questions share the same image).*

## Train/validation leakage
- VQA: TRAIN sample IDs ∩ VAL sample IDs = 0
- VQA: TRAIN image references ∩ VAL image references = 0
- Caption: TRAIN sample IDs ∩ VAL sample IDs = 0
- Caption: TRAIN image references ∩ VAL image references = 0
- Grounding: TRAIN sample IDs ∩ VAL sample IDs = 0
- Grounding: TRAIN image references ∩ VAL image references = 0
- Change VQA: TRAIN sample IDs ∩ VAL sample IDs = 0
- Change VQA: TRAIN pair keys ∩ VAL pair keys = 0
- Change VQA: TRAIN image references ∩ VAL image references = 0

## DataLoader verification
- VQA: `images = torch.Size([2, 3, 224, 224])`
- Caption: `images = torch.Size([2, 3, 224, 224])`
- Grounding: `images = torch.Size([2, 3, 224, 224])`, `bboxes = torch.Size([2, 4])`
- Change VQA: `before_images = torch.Size([2, 3, 224, 224])`, `after_images = torch.Size([2, 3, 224, 224])`

## Pytest results
```
============================= test session starts ==============================
tests/datasets/test_caption_dataset.py::test_caption_dataset_loading PASSED [ 16%]
tests/datasets/test_change_vqa_dataset.py::test_change_vqa_dataset_loading PASSED [ 33%]
tests/datasets/test_dataloaders.py::test_transforms PASSED               [ 50%]
tests/datasets/test_grounding_dataset.py::test_grounding_dataset_loading PASSED [ 66%]
tests/datasets/test_grounding_dataset.py::test_grounding_strict_validation PASSED [ 83%]
tests/datasets/test_vqa_dataset.py::test_vqa_dataset_loading PASSED      [100%]
============================== 6 passed in 3.01s ===============================
```

## BigEarthNet grounding status
BigEarthNet grounding records were fully excluded from Phase 2 due to the unresolved axis semantics. BigEarthNet subsets for grounding are 0.

## Pre-Training Data Cleanup
A localized cleanup was performed on `training_data/subsets/grounding_train.jsonl` to filter out records missing text/query, missing/invalid bbox, or missing image reference without modifying original manifests. The cleaned output was saved to `training_data/subsets/grounding_train_clean.jsonl`.
- Original count: 25590
- Removed count: 2
- Clean count: 25588
- DataLoader result: PASS


## Known limitations
- 2 VRSBench grounding training records are missing text queries.
- Missing BigEarthNet grounding restricts grounding solely to VRSBench limits (25k max vs target 150k).

PHASE2_FINAL_STATUS: PASS
ALL_DATA_LOADERS: PASS
LEAKAGE_CHECK: PASS
DATA_COMPLETENESS: PASS
READY_FOR_PHASE3: YES
