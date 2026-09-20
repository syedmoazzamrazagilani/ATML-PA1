import torch
import torch.nn as nn
import torchvision.models as models
import open_clip

class ModelWrapper(nn.Module):
    def __init__(self, model_name="resnet50", num_classes=10):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        if model_name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2
            backbone = models.resnet50(weights=weights)
            self.feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.norm_mean = [0.485, 0.456, 0.406]
            self.norm_std = [0.229, 0.224, 0.225]
            
        elif model_name == "vit_b_16":
            weights = models.ViT_B_16_Weights.IMAGENET1K_V1
            backbone = models.vit_b_16(weights=weights)
            self.feature_dim = backbone.heads.head.in_features
            backbone.heads = nn.Identity()
            self.backbone = backbone
            self.norm_mean = [0.485, 0.456, 0.406]
            self.norm_std = [0.229, 0.224, 0.225]
            
        elif model_name == "clip_vit_b_32":
            clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
            self.backbone = clip_model.visual
            self.full_clip = clip_model
            self.feature_dim = 512
            self.norm_mean = [0.48145466, 0.4578275, 0.40821073]
            self.norm_std = [0.26862954, 0.26130258, 0.27577711]
            
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.head = nn.Linear(self.feature_dim, num_classes)
        
    def extract_features(self, x):
        with torch.no_grad():
            if self.model_name == "resnet50":
                feat = self.backbone(x)
            elif self.model_name == "vit_b_16":
                feat = self.backbone(x)  # torchvision ViT outputs the class token
            elif self.model_name == "clip_vit_b_32":
                feat = self.backbone(x)
                feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat

    def forward(self, x):
        feat = self.extract_features(x)
        logits = self.head(feat)
        return logits, feat
