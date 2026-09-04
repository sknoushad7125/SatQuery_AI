# Phase 2 Leakage Audit

- **Train Sample IDs**: 213187
- **Val Sample IDs**: 72679
- **Duplicate SIDs within splits**: 0
- **Sample ID Leakage (Train ∩ Val)**: 0
- **Image Leakage (Train ∩ Val)**: 0

## Documentation of Splits
- **VRSBench**: Official Train/Val zip separation is preserved.
- **RSVQA**: Internal split by image ID preserved.
- **CDVQA**: Internal split mapped successfully without overlap.
- **BigEarthNet Grounding**: Excluded from initial subset as axis semantics are unresolved.

**LEAKAGE_CHECK**: PASS
