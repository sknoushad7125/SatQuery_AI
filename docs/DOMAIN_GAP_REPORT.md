# CHANGE DETECTION DOMAIN GAP REPORT

## Current Baseline
The implemented change detection network (`SiamUNet_conc`) is strictly configured for **Optical** bi-temporal change detection (e.g., Cartosat-2S to Cartosat-2S) based on its structural alignment with the LEVIR-CD optical dataset structure.

## Domain Gap Analysis (SIH26167 Requirements)
The final SIH26167 evaluation may include **RISAT SAR** imagery for temporal analysis. 
Applying an optical-trained SiamUNet to SAR imagery or Optical-SAR cross-modal pairs introduces an insurmountable domain gap:
1. **Speckle Noise**: SAR contains multiplicative speckle noise which standard optical convolutions interpret as structural differences, leading to 90%+ False Positive Rates.
2. **Backscatter vs. Reflectance**: SAR measures microwave backscatter (affected by dielectric properties and surface roughness), whereas optical measures solar reflectance. The physical phenomena are incomparable at a raw pixel level.

## Mitigations Implemented
To avoid fabricating performance:
1. `RealSemanticChangeTool.validate()` has been updated to explicitly REJECT any image pair where `modality_1 != modality_2`. Cross-modal temporal comparisons will now cleanly fail rather than silently producing garbage.
2. If both images are SAR, the tool accepts them, but it relies on the zero-shot DINOv2 baseline (semantic embeddings) since the SiamUNet weights are exclusively tuned for optical LEVIR-CD.

For true cross-modal change detection in the final competition, a dedicated heterogenous translation network (e.g., CycleGAN mapping SAR to Optical before differencing) or a joint-embedding space (like CLIP) is required.
