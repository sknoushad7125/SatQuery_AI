import torch
import torch.nn as nn
import timm

class SimpleRSVQAModel(nn.Module):
    def __init__(self, vocab_size, num_answers, embed_dim=128, hidden_dim=256):
        super().__init__()
        # Vision Encoder (frozen mostly for speed, or trainable if needed)
        self.vision = timm.create_model('resnet18', pretrained=True, num_classes=0)
        self.vis_proj = nn.Linear(512, hidden_dim)
        
        # Text Encoder
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_answers)
        )
        
    def forward(self, images, text_seq, text_lens):
        # images: [B, 3, H, W]
        # text_seq: [B, L]
        # text_lens: [B]
        
        v_feat = self.vision(images) # [B, 512]
        v_feat = self.vis_proj(v_feat) # [B, hidden_dim]
        
        emb = self.embedding(text_seq)
        
        # Pack sequence (handle CPU/GPU nicely)
        packed = nn.utils.rnn.pack_padded_sequence(emb, text_lens.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        q_feat = h_n[-1] # [B, hidden_dim]
        
        combined = torch.cat([v_feat, q_feat], dim=1) # [B, hidden_dim*2]
        logits = self.classifier(combined)
        return logits
