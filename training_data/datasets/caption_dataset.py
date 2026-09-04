from .base_dataset import BaseVisionDataset

class CaptionDataset(BaseVisionDataset):
    def __getitem__(self, idx):
        rec = self.records[idx]

        if rec.get("requires_external_imagery", False):
            raise FileNotFoundError(f"Missing external imagery for sample {rec.get('sample_id')}")

        img = self._load_img(rec["image"])
        if self.transforms:
            img = self.transforms(img)

        return {
            "image": img,
            "caption": rec["caption"],
            "sample_id": rec.get("sample_id")
        }
