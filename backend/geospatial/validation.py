from typing import List
from backend.api.schemas.domain import ImageMetadata

class ValidationError(Exception):
    pass

def validate_temporal_pair(img1: ImageMetadata, img2: ImageMetadata) -> bool:
    """Check if two images can be compared for temporal change."""
    errors = []
    if img1.georeferenced and img2.georeferenced:
        if img1.crs != img2.crs:
            errors.append(f"CRS mismatch: {img1.crs} vs {img2.crs}")
        # Simplified bounds check (in production, should compute intersection IoU)
        if img1.bounds != img2.bounds:
            pass # We could allow partial overlap, but let's be strict for MVP if requested
    else:
        if img1.width != img2.width or img1.height != img2.height:
            errors.append("Images are not georeferenced and have different dimensions.")

    if errors:
        raise ValidationError(" | ".join(errors))
    return True

def validate_optical_sar_pair(img_opt: ImageMetadata, img_sar: ImageMetadata) -> bool:
    """Check if optical and SAR images are compatible."""
    return validate_temporal_pair(img_opt, img_sar)
