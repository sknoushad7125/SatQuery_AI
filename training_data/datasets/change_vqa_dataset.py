from .base_dataset import BaseVisionDataset

class ChangeVQADataset(BaseVisionDataset):
    def __getitem__(self, idx):
        rec = self.records[idx]

        before_img = self._load_img(rec["before_image"])
        after_img = self._load_img(rec["after_image"])

        if self.transforms:
            before_img = self.transforms(before_img)
            after_img = self.transforms(after_img)

        return {
            "before_image": before_img,
            "after_image": after_img,
            "question": rec["question"],
            "answer": rec["answer"],
            "sample_id": rec.get("sample_id")
        }
