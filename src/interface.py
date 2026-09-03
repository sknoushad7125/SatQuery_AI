import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.models.baseline import SiameseChangeDetector

class ChangeDetector:
    def __init__(self, checkpoint_path, encoder_name="resnet18", device=None):
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.model = SiameseChangeDetector(encoder_name=encoder_name, pretrained=False)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], additional_targets={'image0': 'image'})
        
    def predict(self, image_a_path, image_b_path):
        """
        Predicts change between two images.
        Input: paths to image A and image B
        Output: dict with change_mask, change_regions, confidence, summary_features
        """
        img_a = np.array(Image.open(image_a_path).convert("RGB"))
        img_b = np.array(Image.open(image_b_path).convert("RGB"))
        
        transformed = self.transform(image=img_a, image0=img_b)
        t_img_a = transformed["image"].unsqueeze(0).to(self.device)
        t_img_b = transformed["image0"].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.model(t_img_a, t_img_b)
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).cpu().numpy()[0, 0]
            
        changed_pixels = np.sum(preds)
        total_pixels = preds.size
        
        # Simple region extraction via connected components or bounding boxes (can use OpenCV)
        # For baseline, we just provide the binary mask and pixel stats
        
        # Confidence can be mean probability of changed pixels
        conf = probs[0, 0].cpu().numpy()
        confidence = float(np.mean(conf[preds > 0])) if changed_pixels > 0 else float(np.max(conf))
        
        return {
            "change_mask": preds.astype(np.uint8), # Binary mask (H, W)
            "change_regions": [], # Bounding boxes to be implemented with cv2
            "confidence": confidence,
            "summary_features": {
                "changed_pixel_count": int(changed_pixels),
                "total_pixels": int(total_pixels),
                "changed_ratio": float(changed_pixels / total_pixels)
            }
        }
