from .base_dataset import BaseVisionDataset

class GroundingDataset(BaseVisionDataset):
    def __getitem__(self, idx):
        rec = self.records[idx]

        if rec.get("requires_external_imagery", False):
            raise FileNotFoundError(f"Missing external imagery for sample {rec.get('sample_id')}")

        img = self._load_img(rec["image"])
        if self.transforms:
            img = self.transforms(img)

        bbox = rec["bbox"]
        # Strict validation
        if len(bbox) != 4: raise ValueError(f"Invalid bbox len: {len(bbox)}")
        if not all(0 <= v <= 1 for v in bbox): raise ValueError("bbox out of bounds")
        x1, y1, x2, y2 = bbox
        if x1 > x2 or y1 > y2: raise ValueError(f"Invalid bbox coords: {bbox}")

        return {
            "image": img,
            "query": rec["text"],
            "bbox": bbox,
            "sample_id": rec.get("sample_id")
        }
