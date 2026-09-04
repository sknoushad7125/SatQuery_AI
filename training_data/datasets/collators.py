import torch

class VQACollator:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        images = torch.stack([item["image"] for item in batch])
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]
        ids = [item["sample_id"] for item in batch]

        return {
            "images": images,
            "questions": questions,
            "answers": answers,
            "sample_ids": ids
        }

class CaptionCollator:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        images = torch.stack([item["image"] for item in batch])
        captions = [item["caption"] for item in batch]
        ids = [item["sample_id"] for item in batch]

        return {
            "images": images,
            "captions": captions,
            "sample_ids": ids
        }

class GroundingCollator:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        images = torch.stack([item["image"] for item in batch])
        queries = [item["query"] for item in batch]
        bboxes = torch.tensor([item["bbox"] for item in batch], dtype=torch.float32)
        ids = [item["sample_id"] for item in batch]

        return {
            "images": images,
            "queries": queries,
            "bboxes": bboxes,
            "sample_ids": ids
        }

class ChangeVQACollator:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        before_images = torch.stack([item["before_image"] for item in batch])
        after_images = torch.stack([item["after_image"] for item in batch])
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]
        ids = [item["sample_id"] for item in batch]

        return {
            "before_images": before_images,
            "after_images": after_images,
            "questions": questions,
            "answers": answers,
            "sample_ids": ids
        }
