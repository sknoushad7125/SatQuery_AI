import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
import io
import base64
from PIL import Image
from typing import List

from backend.tools.base import RemoteSensingTool
from backend.api.schemas.domain import ImageMetadata, ToolResult

class OpticalSARFusionNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        import torchvision.models as models

        self.opt_encoder = models.resnet18(weights=None)
        self.sar_encoder = models.resnet18(weights=None)

        # Adapt optical encoder for 13 channels
        original_opt_conv = self.opt_encoder.conv1
        self.opt_encoder.conv1 = nn.Conv2d(13, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.opt_encoder.conv1.weight[:, :3] = original_opt_conv.weight
            self.opt_encoder.conv1.weight[:, 3:] = 0.0

        # Adapt SAR encoder for 2 channels
        original_sar_conv = self.sar_encoder.conv1
        self.sar_encoder.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.sar_encoder.conv1.weight[:] = original_sar_conv.weight.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1)

        self.fuse1 = nn.Conv2d(64 + 64, 64, kernel_size=1)
        self.fuse2 = nn.Conv2d(128 + 128, 128, kernel_size=1)
        self.fuse3 = nn.Conv2d(256 + 256, 256, kernel_size=1)
        self.fuse4 = nn.Conv2d(512 + 512, 512, kernel_size=1)

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def _forward_encoder(self, encoder, x):
        features = []
        x = encoder.conv1(x)
        x = encoder.bn1(x)
        x = encoder.relu(x)
        x = encoder.maxpool(x)
        x = encoder.layer1(x); features.append(x)
        x = encoder.layer2(x); features.append(x)
        x = encoder.layer3(x); features.append(x)
        x = encoder.layer4(x); features.append(x)
        return features

    def forward(self, opt_x, sar_x):
        opt_feats = self._forward_encoder(self.opt_encoder, opt_x)
        sar_feats = self._forward_encoder(self.sar_encoder, sar_x)

        f1 = self.fuse1(torch.cat([opt_feats[0], sar_feats[0]], dim=1))
        f2 = self.fuse2(torch.cat([opt_feats[1], sar_feats[1]], dim=1))
        f3 = self.fuse3(torch.cat([opt_feats[2], sar_feats[2]], dim=1))
        f4 = self.fuse4(torch.cat([opt_feats[3], sar_feats[3]], dim=1))

        d4 = self.upconv4(f4)
        d4 = torch.cat([d4, f3], dim=1)
        d4 = self.dec_conv4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, f2], dim=1)
        d3 = self.dec_conv3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, f1], dim=1)
        d2 = self.dec_conv2(d2)

        d1 = self.upconv1(d2)
        d1 = self.dec_conv1(d1)

        logits = self.final_conv(d1)
        return logits


class RealDecisionFusionTool(RemoteSensingTool):
    name = "FeatureLevelOpticalSARFusion"
    version = "3.0.0"
    task_types = ["optical_sar_analysis"]
    modalities = ["optical", "sar"]
    execution_type = "real"
    classes = ["vegetation", "built-up area", "water body", "bare land"]

    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.training_status = "trained_checkpoint_loaded"
        self.dataset = "SEN12MS"

    def _load_model(self):
        if self.model is None:
            self.model = OpticalSARFusionNet(num_classes=len(self.classes)).to(self.device)

            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
            fusion_path = os.path.join(project_root, "training", "checkpoints", "optical_sar_fullval_baseline.pth")

            if not os.path.exists(fusion_path):
                raise FileNotFoundError(f"Baseline fusion checkpoint missing at {fusion_path}")

            ckpt = torch.load(fusion_path, map_location=self.device)
            if 'model_state_dict' in ckpt:
                self.model.load_state_dict(ckpt['model_state_dict'])
            else:
                self.model.load_state_dict(ckpt)

            self.model.eval()

    def validate(self, images: List[ImageMetadata], query: str) -> bool:
        if len(images) != 2: return False
        mods = [img.modality for img in images]
        return "optical" in mods and "sar" in mods

    def run(self, images: List[str], query: str, **kwargs) -> ToolResult:
        start_time = time.time()
        try:
            self._load_model()

            # Identify optical and sar image paths
            opt_path, sar_path = images
            if "sar_" in opt_path.lower() or "s1_" in opt_path.lower():
                opt_path, sar_path = sar_path, opt_path

            # Load images
            import rasterio

            with rasterio.open(opt_path) as src:
                opt_data = src.read()
            with rasterio.open(sar_path) as src:
                sar_data = src.read()

            # Convert to float and reshape
            opt_data = opt_data.astype(np.float32)
            sar_data = sar_data.astype(np.float32)

            opt_data = np.clip(opt_data, 0, 10000) / 10000.0
            sar_data = np.clip(sar_data, -25.0, 0.0) / -25.0

            opt_tensor = torch.from_numpy(opt_data).unsqueeze(0).to(self.device)
            sar_tensor = torch.from_numpy(sar_data).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(opt_tensor, sar_tensor)
                probs = torch.softmax(logits, dim=1)
                conf, preds = torch.max(probs, dim=1)

            preds_np = preds.cpu().numpy()[0]
            conf_np = conf.cpu().numpy()[0]

            total_pixels = preds_np.size

            counts = {}
            for i, c_name in enumerate(self.classes):
                c_pixels = np.sum(preds_np == i)
                counts[c_name] = float((c_pixels / total_pixels) * 100.0)

            text = f"Optical-SAR Semantic Segmentation complete. Detected layout: {counts['vegetation']:.1f}% Vegetation, {counts['built-up area']:.1f}% Built-Up Area, {counts['water body']:.1f}% Water Body, {counts['bare land']:.1f}% Bare Land."

            color_map = np.array([
                [34, 139, 34],
                [220, 20, 60],
                [30, 144, 255],
                [218, 165, 32]
            ], dtype=np.uint8)
            rgb_mask = color_map[preds_np]
            mask_img = Image.fromarray(rgb_mask)
            buffered = io.BytesIO()
            mask_img.save(buffered, format="PNG")
            mask_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            return ToolResult(
                tool_name=self.name,
                model_name="DualUNet-FeatureFusion",
                model_type=self.execution_type,
                status="success",
                text=text,
                structured_data={"class_percentages": counts},
                confidence=float(np.mean(conf_np)),
                evidence={"prediction_mask_b64": mask_b64},
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={
                    "status": self.training_status,
                    "checkpoint": "optical_sar_fullval_baseline.pth"
                }
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                model_name="DualUNet-FeatureFusion",
                model_type=self.execution_type,
                status="failure",
                error_message=str(e),
                execution_time_ms=(time.time() - start_time) * 1000
            )
