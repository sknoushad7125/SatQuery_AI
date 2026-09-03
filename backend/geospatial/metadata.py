import rasterio
from typing import Tuple, Dict, Any, Optional
import os
from datetime import datetime

from backend.api.schemas.domain import ImageMetadata

def extract_metadata(filepath: str, file_obj=None) -> ImageMetadata:
    """Extract metadata from an image file (GeoTIFF/TIFF, PNG, JPEG)."""
    _, ext = os.path.splitext(filepath)
    ext = ext.lower()
    
    filename = os.path.basename(filepath)
    format_str = ext.lstrip('.')
    
    if ext in ['.tif', '.tiff']:
        try:
            with rasterio.open(filepath) as src:
                width = src.width
                height = src.height
                bands = src.count
                crs = src.crs.to_string() if src.crs else None
                transform = list(src.transform) if src.transform else None
                bounds = list(src.bounds) if src.bounds else None
                
                # Try to guess modality if possible or leave unknown
                modality = "unknown"
                if bands >= 3:
                    modality = "optical"
                elif bands == 1:
                    modality = "sar" # Naive guess for demo
                
                return ImageMetadata(
                    filename=filename,
                    format=format_str,
                    width=width,
                    height=height,
                    bands=bands,
                    modality=modality,
                    crs=crs,
                    transform=transform,
                    bounds=bounds,
                    georeferenced=bool(crs and bounds)
                )
        except Exception as e:
            raise ValueError(f"Failed to read GeoTIFF metadata: {e}")
            
    elif ext in ['.png', '.jpg', '.jpeg']:
        try:
            from PIL import Image
            with Image.open(filepath) as img:
                width, height = img.size
                bands = len(img.getbands())
                return ImageMetadata(
                    filename=filename,
                    format=format_str,
                    width=width,
                    height=height,
                    bands=bands,
                    modality="optical",
                    georeferenced=False
                )
        except Exception as e:
            raise ValueError(f"Failed to read image metadata: {e}")
    else:
        raise ValueError(f"Unsupported file format: {ext}")
