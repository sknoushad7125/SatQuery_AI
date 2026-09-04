import os
import zipfile
from io import BytesIO
from PIL import Image

class ImageLoader:
    def __init__(self):
        self.zip_handlers = {}

    def load_image(self, image_ref):
        if isinstance(image_ref, dict):
            # ZIP format (VRSBench)
            archive = image_ref.get("archive")
            member = image_ref.get("member")
            if not archive or not member:
                raise ValueError(f"Invalid image dict: {image_ref}")

            if archive not in self.zip_handlers:
                if not os.path.exists(archive):
                    raise FileNotFoundError(f"Archive not found: {archive}")
                self.zip_handlers[archive] = zipfile.ZipFile(archive, 'r')

            try:
                img_data = self.zip_handlers[archive].read(member)
                img = Image.open(BytesIO(img_data)).convert('RGB')
                return img
            except KeyError:
                raise FileNotFoundError(f"Member {member} not found in {archive}")

        elif isinstance(image_ref, str):
            # Local file path
            if not os.path.exists(image_ref):
                raise FileNotFoundError(f"Local image not found: {image_ref}")
            return Image.open(image_ref).convert('RGB')
        else:
            raise TypeError(f"Unknown image reference type: {type(image_ref)}")

    def close(self):
        for z in self.zip_handlers.values():
            z.close()
        self.zip_handlers.clear()
