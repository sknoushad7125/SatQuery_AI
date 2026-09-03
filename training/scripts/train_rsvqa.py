import argparse
import os
import sys
import time
import torch
from torch.utils.data import Dataset
from transformers import BlipProcessor, BlipForQuestionAnswering, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import numpy as np
from PIL import Image
import json

class RSVQADataset(Dataset):
    def __init__(self, data_dir, split, processor):
        self.data_dir = data_dir
        self.split = split
        self.processor = processor
        
        q_path = os.path.join(data_dir, f"{split}_questions.json")
        if not os.path.exists(q_path):
            raise FileNotFoundError(f"RSVQA dataset structure not found in {data_dir}")
            
        with open(q_path, "r") as f:
            self.data = json.load(f)["questions"]
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.data_dir, "images", f"{item['img_id']}.tif")
        img = Image.open(img_path).convert("RGB")
        q = item["question"]
        a = item["answer"]
        
        inputs = self.processor(images=img, text=q, return_tensors="pt", padding="max_length", max_length=32)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        labels = self.processor.tokenizer(a, return_tensors="pt", padding="max_length", max_length=16).input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs

class MockRSVQADataset(Dataset):
    def __init__(self, num_samples, processor):
        self.num_samples = num_samples
        self.processor = processor
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        img = Image.new("RGB", (224, 224), color=tuple(np.random.randint(0, 255, 3).tolist()))
        q = "Is there a building?"
        a = "yes"
        inputs = self.processor(images=img, text=q, return_tensors="pt", padding="max_length", max_length=32)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        labels = self.processor.tokenizer(a, return_tensors="pt", padding="max_length", max_length=16).input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/app/datasets/rsvqa")
    parser.add_argument("--output_dir", type=str, default="/app/training/checkpoints/rsvqa_lora")
    parser.add_argument("--model_name", type=str, default="Salesforce/blip-vqa-base")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--synthetic-dev-mode", action="store_true")
    args = parser.parse_args()

    if not args.synthetic_dev_mode and not os.path.exists(args.dataset_path):
        print("Dataset not found.\n")
        print("Training cannot proceed because a real remote-sensing dataset")
        print("is required for SIH26167 validation.\n")
        print("See docs/DATASET_SETUP.md")
        sys.exit(1)

    processor = BlipProcessor.from_pretrained(args.model_name)
    model = BlipForQuestionAnswering.from_pretrained(args.model_name)
    
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none"
    )
    model = get_peft_model(model, config)
    
    if args.synthetic_dev_mode:
        print("WARNING: Using DEV_TEST_ONLY synthetic dataset.")
        train_dataset = MockRSVQADataset(64, processor)
        val_dataset = MockRSVQADataset(16, processor)
        dataset_name = "Synthetic_DEV_TEST_ONLY"
    else:
        print(f"Loading real RSVQA dataset from {args.dataset_path}")
        train_dataset = RSVQADataset(args.dataset_path, "train", processor)
        val_dataset = RSVQADataset(args.dataset_path, "val", processor)
        dataset_name = "RSVQA"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        save_strategy="epoch",
        eval_strategy="epoch",
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    trainer.train()
    model.save_pretrained(args.output_dir)
    print(f"RSVQA LoRA adapter saved to {args.output_dir}")
    
    metadata = {
        "dataset_name": dataset_name,
        "sample_counts": {"train": len(train_dataset), "val": len(val_dataset)},
        "training_configuration": vars(args),
        "validation_metric": "Loss monitoring via Trainer",
        "training_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint_path": args.output_dir
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    main()
