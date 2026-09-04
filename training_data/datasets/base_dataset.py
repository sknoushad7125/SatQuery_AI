import json
from torch.utils.data import Dataset
from .image_loader import ImageLoader

class BaseVisionDataset(Dataset):
    def __init__(self, jsonl_path, transforms=None, tokenizer=None):
        self.records = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.records.append(json.loads(line))

        self.transforms = transforms
        self.tokenizer = tokenizer
        self.loader = ImageLoader()

    def __len__(self):
        return len(self.records)

    def _load_img(self, img_ref):
        return self.loader.load_image(img_ref)

    def __del__(self):
        if hasattr(self, 'loader'):
            self.loader.close()
