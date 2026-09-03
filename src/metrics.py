import torch

class ChangeDetectionMetrics:
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        
    def update(self, preds, targets):
        """
        preds: [B, 1, H, W] - probabilities or binary masks
        targets: [B, 1, H, W] - binary masks
        """
        if preds.dtype != torch.bool and preds.is_floating_point():
            preds = (preds > 0.5)
            
        preds = preds.bool()
        targets = targets.bool()
        
        self.tp += (preds & targets).sum().item()
        self.tn += (~preds & ~targets).sum().item()
        self.fp += (preds & ~targets).sum().item()
        self.fn += (~preds & targets).sum().item()
        
    def compute(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        iou = self.tp / (self.tp + self.fp + self.fn + 1e-8)
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + 1e-8)
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "accuracy": accuracy,
            "confusion_matrix": {
                "tp": self.tp,
                "tn": self.tn,
                "fp": self.fp,
                "fn": self.fn
            }
        }
        
    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
