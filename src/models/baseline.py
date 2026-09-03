import torch
import torch.nn as nn
import timm

class SiameseChangeDetector(nn.Module):
    def __init__(self, encoder_name='resnet18', pretrained=True):
        super().__init__()
        # Shared Encoder
        self.encoder = timm.create_model(encoder_name, pretrained=pretrained, features_only=True)
        
        # We find the channel dimension of the highest resolution feature map (typically stage 1 or 2)
        # But for a simple baseline, let's extract the final feature map and upsample.
        # Even better: simple decoder matching UNet style, or just a linear classification head over difference.
        
        # Let's get encoder channels
        dummy = torch.zeros(1, 3, 256, 256)
        features = self.encoder(dummy)
        
        self.decoders = nn.ModuleList()
        # Create a lightweight decoder: upsample + conv for each stage, summing them up
        decoder_channels = 128
        self.final_conv = nn.Sequential(
            nn.Conv2d(decoder_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )
        
        self.convs = nn.ModuleList([
            nn.Conv2d(f.shape[1], decoder_channels, 1) for f in features
        ])

    def forward(self, img_a, img_b):
        feat_a = self.encoder(img_a)
        feat_b = self.encoder(img_b)
        
        # Absolute difference
        diffs = [torch.abs(a - b) for a, b in zip(feat_a, feat_b)]
        
        # Simple Feature Pyramid upsampling
        out = None
        for diff, conv in zip(reversed(diffs), reversed(self.convs)):
            x = conv(diff)
            if out is None:
                out = x
            else:
                out = x + torch.nn.functional.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
                
        # Upsample to original image size
        out = torch.nn.functional.interpolate(out, size=img_a.shape[2:], mode='bilinear', align_corners=False)
        out = self.final_conv(out)
        
        return out
