import argparse
import os
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import numpy as np
import evaluate

def compute_metrics(eval_pred):
    metric = evaluate.load("f1")
    logits, labels = eval_pred
    predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    return metric.compute(predictions=predictions, references=labels, average="micro")

def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapter on BigEarthNet")
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--output_dir", type=str, default="../checkpoints/bigearthnet_lora")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1, help="For quick testing")
    args = parser.parse_args()

    print(f"Loading Base Model: {args.model_name}")
    # Using 43 classes for BigEarthNet
    model = AutoModelForImageClassification.from_pretrained(
        args.model_name,
        num_labels=43,
        ignore_mismatched_sizes=True
    )
    
    print("Configuring LoRA...")
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query", "value"], # For ViT attention
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"]
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    print("Loading BigEarthNet Dataset (this may take time if downloading...)")
    # For a real pipeline, we use the HF datasets loader for bigearthnet
    # Note: BigEarthNet requires manual download in some cases, so we support a mock dataset load for testing
    try:
        # Load a small subset for quick training/testing
        dataset = load_dataset("Bingsu/BigEarthNet_19_classes", split="train[:1%]")
        eval_dataset = load_dataset("Bingsu/BigEarthNet_19_classes", split="validation[:1%]")
        # Note: Bingsu's dataset has 19 classes, if using original we'd need 43. 
        # For this prototype we will adjust labels dynamically based on dataset.
        num_labels = len(dataset.features['labels'].feature.names)
        model.classifier = torch.nn.Linear(model.config.hidden_size, num_labels)
    except Exception as e:
        print(f"Failed to load dataset: {e}. Ensure you have network access.")
        return

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    
    def preprocess_images(examples):
        return processor(examples["image"], return_tensors="pt")
        
    def preprocess_labels(examples):
        # Multi-hot encode labels
        labels_batch = []
        for labels in examples["labels"]:
            multi_hot = np.zeros(num_labels, dtype=np.float32)
            multi_hot[labels] = 1.0
            labels_batch.append(multi_hot)
        return {"labels": labels_batch}

    # Apply preprocessing
    print("Preprocessing datasets...")
    train_dataset = dataset.map(preprocess_images, batched=True, remove_columns=["image", "name"])
    train_dataset = train_dataset.map(preprocess_labels, batched=True)
    
    eval_dataset = eval_dataset.map(preprocess_images, batched=True, remove_columns=["image", "name"])
    eval_dataset = eval_dataset.map(preprocess_labels, batched=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        remove_unused_columns=False,
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=5e-4,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=10,
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )
    
    print("Starting Training...")
    trainer.train()
    
    print(f"Saving fine-tuned LoRA model to {args.output_dir}")
    model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
