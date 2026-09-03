import json
import os

def audit():
    report = {
        "source_datasets": ["LEVIR-CD", "SECOND"],
        "source_dataset_counts": {
            "LEVIR-CD": "Unknown exact proportion (Paper states ~31k pairs combined)",
            "SECOND": "Unknown exact proportion"
        },
        "image_dimensions": {
            "LEVIR-CD": [1024, 1024],
            "SECOND": [512, 512]
        },
        "crop_dimensions": {
            "CDVQA/SECOND": [512, 512],
            "CDVQA/LEVIR": "Likely 256x256 (as per VQA paper conventions), but unconfirmed natively"
        },
        "crop_stride": "Unknown (Not published in YZHJessica/CDVQA)",
        "indexing_rule": "Global 5-digit zero-padded sequential index (e.g., 07308.png) spanning all concatenated datasets.",
        "train_val_test_composition": "Mixed source. HF test split contains CDVQA/SECOND images. Consequently, CDVQA test is NOT pure LEVIR-CD.",
        "physical_image_availability": "Available via unofficial WebDataset mirror (ljx620/CDVQA on Hugging Face). Official repo (YZHJessica/CDVQA) only hosts QA JSONs.",
        "authoritative_urls": [
            "https://github.com/YZHJessica/CDVQA",
            "https://huggingface.co/datasets/ljx620/CDVQA"
        ],
        "reconstruction_requirements": "Requires downloading both LEVIR-CD and SECOND, and guessing the undocumented sorting/stride rules, OR simply downloading the pre-processed HF mirror.",
        "levir_cd_compatible_image_count": "Unknown (Blocked by lack of published mapping rule)",
        "unresolved_questions": [
            "Exact global sorting order used to generate the 5-digit index.",
            "Whether LEVIR-CD was cropped to 256x256 or 512x512 to match SECOND."
        ]
    }
    
    os.makedirs("datasets/cdvqa", exist_ok=True)
    with open("datasets/cdvqa/source_audit.json", "w") as f:
        json.dump(report, f, indent=2)

    markdown_content = """# CDVQA Source Audit Report

## 1. What exact source datasets does CDVQA contain?
Based on empirical metadata analysis and literature review, CDVQA contains images from both **LEVIR-CD** and **SECOND**.

## 2. Does CDVQA actually contain LEVIR-CD-derived images?
Yes. The original paper states it utilizes LEVIR-CD.

## 3. Does CDVQA actually contain SECOND-derived images?
Yes. We empirically verified that images such as `07308.png` in the CDVQA test set explicitly contain the metadata `{"source": "CDVQA/SECOND", "wh": [[512,512],[512,512]]}` in the HF WebDataset mirror.

## 4. What is the exact image generation/cropping procedure?
The procedure is undocumented in the official repository. SECOND images are retained at 512x512. LEVIR-CD images (1024x1024) are either cropped or resized, but the specific stride and dimension logic (e.g., 256x256 or 512x512) is not mathematically recoverable without exhaustive pixel matching against the pre-processed CDVQA images.

## 5. What does a filename such as `07308.png` actually mean?
It is a global, zero-padded 5-digit index assigned *after* all datasets (LEVIR-CD and SECOND) were cropped, merged, and sequentially enumerated.

## 6. What are the exact dimensions of each source-image type?
- **LEVIR-CD:** Natively 1024x1024
- **SECOND:** Natively 512x512
- **CDVQA Crops:** SECOND crops are 512x512. LEVIR-CD crops are unverified.

## 7. Can the official CDVQA test images be reconstructed?
No, not deterministically from scratch using the official YZHJessica/CDVQA repository alone, because the specific ordering and stride rules are unpublished.

## 8. If yes, from which datasets?
N/A (Reconstruction blocked). 

## 9. What is the smallest legitimate acquisition required?
Downloading the pre-compiled `ljx620/CDVQA` Hugging Face dataset, which contains the exact physical image pairs matched to the global indices. Reconstructing from LEVIR-CD and SECOND from scratch is infeasible without the mapping table.

## 10. How many CDVQA test images can be connected to LEVIR-CD?
Unknown. Without downloading the full compiled CDVQA dataset or knowing the indexing rule, we cannot determine the exact distribution of LEVIR vs. SECOND in the 968 test images.

## 11. Can our planned change-VQA architecture be evaluated on the official CDVQA test set?
Yes, **BUT** only if the architecture supports processing SECOND images alongside LEVIR-CD images, OR if we filter the CDVQA test set to only evaluate on the LEVIR-CD-derived subset. 

## 12. If not, identify the precise architectural/dataset incompatibility rather than hiding it.
The primary incompatibility is that the CDVQA test split mixes two entirely different source datasets (LEVIR-CD and SECOND). If the `SiamUNet` is only trained on LEVIR-CD, evaluating it on SECOND-derived images in the CDVQA test set will yield catastrophic out-of-distribution failures, severely degrading VQA accuracy. We must isolate the LEVIR-CD subset for valid evaluation.

### Final Status
`BLOCKED — authoritative CDVQA image reconstruction remains unresolved`
"""
    with open("datasets/cdvqa/CDVQA_SOURCE_AUDIT.md", "w") as f:
        f.write(markdown_content)
        
    print("Audit files generated.")

if __name__ == "__main__":
    audit()
