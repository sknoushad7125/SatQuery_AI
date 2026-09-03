import torch
import torch.nn as nn
import torch.nn.functional as F

class OpticalSARDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, logits, targets):
        """
        logits: [B, C, H, W] raw unnormalized scores
        targets: [B, H, W] integer class indices
        """
        probs = F.softmax(logits, dim=1)
        # Convert targets to one-hot: [B, H, W, C] -> [B, C, H, W]
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        
        # Calculate intersection and cardinality per class across spatial and batch dimensions
        intersection = torch.sum(probs * targets_one_hot, dim=(0, 2, 3))
        cardinality = torch.sum(probs + targets_one_hot, dim=(0, 2, 3))
        
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        
        # Return 1 - mean dice over all classes
        return 1.0 - torch.mean(dice_score)

class OpticalSARLoss(nn.Module):
    def __init__(self, class_weights, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        # Ensure class_weights is passed to CE loss to penalize minority class errors
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = OpticalSARDiceLoss()
        self.ce_w = ce_weight
        self.dice_w = dice_weight
        
    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.ce_w * ce + self.dice_w * dice
