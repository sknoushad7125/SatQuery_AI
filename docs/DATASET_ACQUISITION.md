# DATASET ACQUISITION PROCEDURE

This document specifies the legitimate open-source locations and structures for all datasets required by SatQuery AI.
**DO NOT DOWNLOAD FULL DATASETS FOR DEVELOPMENT.** Use the recommended subsets.

## 1. BigEarthNet.txt (Primary Adaptation Dataset)
* **Official Source**: Defined in arXiv:2603.29630. Extension of TU Berlin's BigEarthNet.
* **Content**: Co-registered Sentinel-1 SAR (VV/VH), Sentinel-2 optical, and diverse text annotations.
* **Download Method**: Follow links in the paper's associated GitHub repository to download the JSON text annotations and the corresponding S1/S2 patches.
* **Target Path**: `datasets/bigearthnet/`
* **Expected Structure**:
  ```
  datasets/bigearthnet/
  ├── annotations.json (Contains image_id and text descriptions)
  ├── images/
      ├── S1_patch_.../
      └── S2_patch_.../
  ```

## 2. LEVIR-CD
* **Official Source**: BUAA LEVIR Lab (https://justcheneng.github.io/LEVIR/)
* **Download Method**: Google Drive or Baidu Pan links provided on their official page.
* **Target Path**: `datasets/levir_cd/`
* **Expected Structure**:
  ```
  datasets/levir_cd/
  ├── train/ (and val/ test/)
      ├── A/ (Time 1 images, .png)
      ├── B/ (Time 2 images, .png)
      └── label/ (Change masks, .png)
  ```
* **Prototype Size**: 50 pairs manually sliced into 256x256 crops.

## 3. SEN12MS
* **Official Source**: TUM (https://mediatum.ub.tum.de/1474000)
* **Download Method**: Web download via MediaTUM. 
* **Target Path**: `datasets/sen12ms/`
* **Expected Structure**:
  ```
  datasets/sen12ms/
  ├── ROIsXXXX_Season/
      ├── s1_X/ (Sentinel-1 SAR)
      ├── s2_X/ (Sentinel-2 Optical)
      └── lc_X/ (MODIS Land Cover)
  ```
* **Prototype Size**: Download a single ROI folder (e.g., ROIs1158_Spring) instead of the 80GB archive.

## 4. RSVQA
* **Official Source**: Sylvain Lobry (rsvqa.sylvainlobry.com)
* **Download Method**: Zenodo.
* **Target Path**: `datasets/rsvqa/`
* **Expected Structure**:
  ```
  datasets/rsvqa/
  ├── Images_LR/ (GeoTIFFs)
  ├── LR_questions.json
  └── LR_answers.json
  ```
* **Prototype Size**: 1,000 LR images and corresponding JSON subset.

## 5. VRSBench
* **Official Source**: VRSBench GitHub releases.
* **Target Path**: `datasets/vrsbench/`
* **Prototype Strategy**: Do not download automatically. Pretrained models (OWL-ViT) handle grounding via agentic orchestration. Keep empty or use tiny 10-image subset for evaluation.

## 6. CDVQA
* **Official Source**: Academic papers derived from LEVIR-CD. 
* **Prototype Strategy**: Because no centralized repository exists comparable to LEVIR-CD, CDVQA queries are generated synthetically from LEVIR-CD masks to demonstrate capability. 
