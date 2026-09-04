# BigEarthNet.txt Bounding Box Semantics

## Source
BigEarthNet.txt dataset (Parquet files), HuggingFace README, official `example_data_loading.py`, `ben_txt_datamodule.py`, and the official paper (arXiv:2603.29630).

## Verified coordinate range
Coordinates are explicitly normalized to [0.0, 1.0].

## Verified coordinate order
The coordinates are arranged as `[min1 min2, max1 max2]`.
Without official documentation, it is ambiguous whether this represents `[ymin, xmin, ymax, xmax]` or `[xmin, ymin, xmax, ymax]`. Both axes always satisfy `min <= max`.

## Coordinate origin
Standard computer vision coordinate system, where (0,0) corresponds to the top-left image corner.

## Corner vs width/height interpretation
The values represent corners `[min, min, max, max]` rather than `[x, y, width, height]`. If it were `w, h`, values like `1.0` in the 4th position would cause `y + h > 1.0`, which violates image boundaries.

## Example annotations
Dataset strings appear as `[0.64 0.0, 1.0 0.71]`.

## Current parser behavior
`prepare_bigearthnet.py` strips punctuation and casts to 4 floats in order: `[0.64, 0.0, 1.0, 0.71]`. It assumes this directly maps to the `[x1, y1, x2, y2]` schema required.

## Required conversion
Because the exact `x` vs `y` mapping cannot be definitively proven from the dataset alone, and assuming `[ymin, xmin, ymax, xmax]` is common in EO models, the current parser risks silently swapping X and Y. We must halt and confirm the axis order.

## Authoritative Verification
- **Source inspected**: `datasets/bigearthnet_txt/README.md`, `ben_txt_datamodule.py`, `example_data_loading.py`, Hugging Face dataset repository metadata, and the official paper (`arXiv:2603.29630`).
- **Exact semantic definition**: Not found. The code handles bounding boxes purely as raw strings (e.g., `output = sample.output`) and passes them directly to VLMs.
- **Evidence**: No explicit mention of `xmin`, `ymin`, or axis ordering exists in the datamodule, preprocessing code, or the paper text when describing referring expression annotations.
- **Empirical cross-check**: Because BigEarthNet.txt relies on external imagery that is not locally downloaded (`image_available: false`), we cannot empirically plot the boxes over the images to visually confirm the axis orientation.
- **Conclusion**: The dataset consistently uses `[min1 min2, max1 max2]`, but which axis corresponds to X and which to Y is genuinely undocumented in the available sources.
- **Modification**: `prepare_bigearthnet.py` should NOT be modified yet, as no conversion rule can be mathematically proven.

BIGEARTHNET_BBOX: UNRESOLVED
BBOX_CONVERSION_REQUIRED: NO
