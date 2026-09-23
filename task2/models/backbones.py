import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        resnet = models.resnet18(weights=weights)
        
        self.feature_dim = resnet.fc.in_features
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x):
        x = self.backbone(x)
        return torch.flatten(x, 1)
        
    def train(self, mode=True):
        """
        CRITICAL: Override train() to enforce the frozen BatchNorm policy.
        The complete model is set to train mode, but BN layers are forced back 
        to eval() so running statistics do not update on source-target mixtures.
        Scale and bias parameters remain trainable.
        """
        super().train(mode)
        if mode:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
                    if module.weight is not None:
                        module.weight.requires_grad_(True)
                    if module.bias is not None:
                        module.bias.requires_grad_(True)

class ClassifierHead(nn.Module):
    def __init__(self, in_features=512, num_classes=7):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.fc(x)
