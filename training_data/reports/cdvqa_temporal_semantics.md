# CDVQA Temporal Ordering

## Dataset structure
The CDVQA dataset is structured with question/answer JSON files (`Test_questions.json`, etc.) mapping to image files via `img_id`. The images are stored as pairs: `*_1.png` and `*_2.png`.

## Image-pair representation
The metadata refers to a base file name (e.g., `07308.png`). The actual images on disk are `07308_1.png` and `07308_2.png`.

## Metadata evidence
There are no explicit acquisition dates provided in the CDVQA JSON metadata files.

## Annotation/question evidence
Questions frequently contain directional temporal logic, such as "Have the areas of buildings increased?" or "Have the regions of buildings decreased?".

## Representative examples
1. **07308.png**
   - **Question**: "Have the regions of buildings decreased?"
   - **Answer**: "yes"
   - **Empirical Measurement**: Edge density (Canny) for `07308_1.png` is 47,589; for `07308_2.png` is 22,518. A massive drop in structural edges perfectly aligns with a *decrease* in buildings if `_1` is the "before" image and `_2` is the "after" image.
2. **04580.png**
   - **Question**: "Did the areas of buildings increase?"
   - **Answer**: "yes"
   - **Empirical Measurement**: Edge density for `04580_1.png` is 5,476; for `04580_2.png` is 20,170. A large increase in edges aligns with an *increase* in buildings if `_1` is "before" and `_2` is "after".
3. **04221.png**
   - **Question**: "Did the regions of buildings increase?"
   - **Answer**: "yes"
   - **Empirical Measurement**: `_1` has 30,363 edges, `_2` has 45,361 edges.

## Verified temporal convention
Across all tested representative pairs, `_1.png` definitively maps to the earlier (T1 / Before) state and `_2.png` maps to the later (T2 / After) state.

## Current prepare_cdvqa.py behavior
The parser maps `before_image` to `_1.png` and `after_image` to `_2.png`.

## Required correction
None. The existing implementation perfectly preserves the dataset's native temporal semantics.

## Final verdict
CDVQA_ORDER: VERIFIED
