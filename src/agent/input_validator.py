import os
from typing import List, Tuple
from PIL import Image
from src.agent.schemas import ImageInput, InputConfiguration, Modality

class InputValidator:
    ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    def validate(self, images: List[ImageInput]) -> Tuple[bool, str, InputConfiguration]:
        if not images:
            return False, "No images provided.", None
            
        if len(images) > 2:
            return False, f"Maximum 2 images supported, received {len(images)}.", None

        formats = []
        dimensions = []
        modality = Modality.UNKNOWN
        
        for img in images:
            ext = os.path.splitext(img.filepath)[1].lower()
            if ext not in self.ALLOWED_EXTENSIONS:
                return False, f"Unsupported extension {ext} for {img.filepath}", None
            if not os.path.exists(img.filepath):
                return False, f"Missing file: {img.filepath}", None
            try:
                with Image.open(img.filepath) as im:
                    dimensions.append(list(im.size))
            except Exception as e:
                return False, f"Corrupt image {img.filepath}: {str(e)}", None
            formats.append(ext)
            
        if len(images) == 2 and dimensions[0] != dimensions[1]:
            return False, "Mismatched dimensions for co-registered pair.", None
            
        temporal_pair = False
        cross_modal_pair = False
        
        if len(images) == 2:
            mod_a = images[0].modality
            mod_b = images[1].modality
            
            if mod_a == Modality.OPTICAL and mod_b == Modality.SAR or mod_a == Modality.SAR and mod_b == Modality.OPTICAL:
                cross_modal_pair = True
                modality = Modality.UNKNOWN # Mixed
            else:
                temporal_pair = True
                modality = mod_a if mod_a != Modality.UNKNOWN else Modality.OPTICAL
        else:
            modality = images[0].modality if images[0].modality != Modality.UNKNOWN else Modality.OPTICAL
            
        config = InputConfiguration(
            number_of_images=len(images),
            modality=modality,
            image_format=formats,
            dimensions=dimensions,
            geospatial_metadata_available=False,
            temporal_pair=temporal_pair,
            cross_modal_pair=cross_modal_pair
        )
        return True, "Valid", config
