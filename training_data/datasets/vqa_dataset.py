from .base_dataset import BaseVisionDataset

class VQADataset(BaseVisionDataset):
    def __getitem__(self, idx):
        rec = self.records[idx]

        # BigEarthNet missing image check
        if rec.get("requires_external_imagery", False):
            raise FileNotFoundError(f"Missing external imagery for BigEarthNet sample {rec.get('sample_id')}")

        img_ref = rec.get("image")
        if not img_ref:
            raise ValueError(f"No image reference for sample {rec.get('sample_id')}")

        img = self._load_img(img_ref)
        if self.transforms:
            img = self.transforms(img)

        q = rec["question"]
        a = rec["answer"]

        # If tokenizer exists, we could tokenize here, but standard practice is collator
        return {
            "image": img,
            "question": q,
            "answer": a,
            "sample_id": rec.get("sample_id")
        }
