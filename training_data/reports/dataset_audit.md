# Dataset Preprocessing Audit

**Validation Status**: PASS

## Statistics
- **Total Records**: 7420465
- **Unique Images (Local + ZIP)**: 30184

### By Dataset
- vrsbench: 194585
- rsvqa: 57223
- bigearthnet_txt: 7128971
- cdvqa: 39686

### By Task
- captioning: 376823
- vqa: 5326040
- grounding: 1677916
- change_vqa: 39686

### By Split
- val: 62918
- internal_val: 5703
- train: 4897154
- validation: 2454690

## Notes
- BigEarthNet raw imagery is correctly flagged as missing/external.
- RSVQA split leakage checked: PASS
- VRSBench split leakage checked: PASS
