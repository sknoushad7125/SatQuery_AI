import torch

class SegmentationMetrics:
    def __init__(self, num_classes=4, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or ["vegetation", "built-up area", "water body", "bare land"]
        self.reset()
        
    def reset(self):
        self.intersections = torch.zeros(self.num_classes)
        self.unions = torch.zeros(self.num_classes)
        self.preds_counts = torch.zeros(self.num_classes)
        self.targets_counts = torch.zeros(self.num_classes)
        self.correct_pixels = 0
        self.total_pixels = 0
        
    def update(self, logits, targets):
        """
        logits: [B, C, H, W]
        targets: [B, H, W]
        """
        preds = torch.argmax(logits, dim=1)
        
        self.correct_pixels += (preds == targets).sum().item()
        self.total_pixels += targets.numel()
        
        for c in range(self.num_classes):
            pred_c = (preds == c)
            target_c = (targets == c)
            
            self.intersections[c] += (pred_c & target_c).sum().item()
            self.unions[c] += (pred_c | target_c).sum().item()
            self.preds_counts[c] += pred_c.sum().item()
            self.targets_counts[c] += target_c.sum().item()
            
    def compute(self):
        eps = 1e-7
        ious = self.intersections / (self.unions + eps)
        dices = 2 * self.intersections / (self.preds_counts + self.targets_counts + eps)
        
        # Only compute macro averages over classes that are present in the target OR prediction
        valid_mask = self.unions > 0
        miou = ious[valid_mask].mean().item() if valid_mask.any() else 0.0
        mdice = dices[valid_mask].mean().item() if valid_mask.any() else 0.0
        
        pixel_acc = self.correct_pixels / self.total_pixels if self.total_pixels > 0 else 0.0
        
        per_class_iou = {self.class_names[c]: ious[c].item() for c in range(self.num_classes)}
        per_class_dice = {self.class_names[c]: dices[c].item() for c in range(self.num_classes)}
        
        return {
            "mIoU": miou,
            "mDice": mdice,
            "pixel_accuracy": pixel_acc,
            "per_class_iou": per_class_iou,
            "per_class_dice": per_class_dice
        }
