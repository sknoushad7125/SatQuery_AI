from typing import List, Tuple, Optional
from backend.api.schemas.domain import ImageMetadata

class QueryRouter:
    def route(self, query: str, images: List[ImageMetadata], input_type: Optional[str] = None) -> Tuple[str, str]:
        q = query.lower()
        num_images = len(images)
        modalities = [img.modality.lower() for img in images]

        # 1. Input Type Validation
        if input_type == "temporal_pair":
            if num_images != 2:
                return "validation_failed", "Temporal pair requires exactly two images."
            if modalities[0] != modalities[1]:
                return "validation_failed", "Temporal pair requires two images of the same modality (e.g. two optical images)."

        elif input_type == "optical_sar_pair":
            if num_images != 2:
                return "validation_failed", "Optical-SAR pair requires exactly two images."
            opt_count = modalities.count("optical") + modalities.count("multispectral")
            sar_count = modalities.count("sar")
            if opt_count != 1 or sar_count != 1:
                return "validation_failed", "Optical-SAR pair requires exactly one optical/multispectral image and one SAR image."

        elif input_type == "single":
            if num_images != 1:
                return "validation_failed", "Single-image analysis requires exactly one image."

        if num_images == 0 or num_images > 2:
            return "validation_failed", "Invalid number of images."

        is_temporal = (input_type == "temporal_pair")
        is_optsar = (input_type == "optical_sar_pair")
        is_single_sar = (num_images == 1 and modalities[0] == "sar")
        is_single_opt = (num_images == 1 and modalities[0] in ["optical", "multispectral"])

        caption_keywords = ["describe", "caption", "scene", "generate a caption"]
        vqa_keywords = ["what", "are there", "is there", "how", "where", "image", "roads", "analyze", "classify", "land cover", "identify", "visible"]

        intent_caption = any(w in q for w in caption_keywords)
        intent_vqa = any(w in q for w in vqa_keywords)

        sar_keywords = ["sar", "radar", "backscatter"]
        opt_keywords = ["optical", "rgb", "multispectral"]
        intent_sar = any(w in q for w in sar_keywords)
        intent_opt = any(w in q for w in opt_keywords)

        # Modality vs Intent Validation
        if is_single_opt and intent_sar:
            return "validation_failed", "Your query requests SAR analysis, but the uploaded image is optical/multispectral. Please upload a SAR image or revise the query."
        if is_single_sar and intent_opt:
            return "validation_failed", "Your query requests optical analysis, but the uploaded image is SAR. Please upload an optical image or revise the query."

        if is_optsar:
            return "optical_sar", "optical-sar pair configuration explicitly takes precedence"

        if is_temporal:
            return "change_detection", "temporal pair configuration explicitly takes precedence"

        if is_single_sar:
            return "sar_classification", "single SAR configuration strictly routes to SAR classification"

        if is_single_opt:
            if intent_caption:
                return "captioning", "single optical + caption intent"
            elif intent_vqa:
                return "vqa", "single optical + VQA intent"
            else:
                return "vqa", "fallback to VQA for general optical query"

        return "unsupported", "unsupported modality combination or configuration"
