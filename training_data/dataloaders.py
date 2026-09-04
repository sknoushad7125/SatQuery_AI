from torch.utils.data import DataLoader
from torchvision import transforms
from .datasets.vqa_dataset import VQADataset
from .datasets.caption_dataset import CaptionDataset
from .datasets.grounding_dataset import GroundingDataset
from .datasets.change_vqa_dataset import ChangeVQADataset
from .datasets.collators import VQACollator, CaptionCollator, GroundingCollator, ChangeVQACollator

def get_default_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # Default ImageNet norm
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_vqa_dataloader(jsonl_path, batch_size=4, num_workers=0, shuffle=True, image_size=224, tokenizer=None, transforms_fn=None):
    tfm = transforms_fn if transforms_fn else get_default_transforms(image_size)
    dataset = VQADataset(jsonl_path, transforms=tfm, tokenizer=tokenizer)
    collator = VQACollator(tokenizer=tokenizer)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, collate_fn=collator)

def create_caption_dataloader(jsonl_path, batch_size=4, num_workers=0, shuffle=True, image_size=224, tokenizer=None, transforms_fn=None):
    tfm = transforms_fn if transforms_fn else get_default_transforms(image_size)
    dataset = CaptionDataset(jsonl_path, transforms=tfm, tokenizer=tokenizer)
    collator = CaptionCollator(tokenizer=tokenizer)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, collate_fn=collator)

def create_grounding_dataloader(jsonl_path, batch_size=4, num_workers=0, shuffle=True, image_size=224, tokenizer=None, transforms_fn=None):
    tfm = transforms_fn if transforms_fn else get_default_transforms(image_size)
    dataset = GroundingDataset(jsonl_path, transforms=tfm, tokenizer=tokenizer)
    collator = GroundingCollator(tokenizer=tokenizer)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, collate_fn=collator)

def create_change_vqa_dataloader(jsonl_path, batch_size=4, num_workers=0, shuffle=True, image_size=224, tokenizer=None, transforms_fn=None):
    tfm = transforms_fn if transforms_fn else get_default_transforms(image_size)
    dataset = ChangeVQADataset(jsonl_path, transforms=tfm, tokenizer=tokenizer)
    collator = ChangeVQACollator(tokenizer=tokenizer)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, collate_fn=collator)
