import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        resnet = models.resnet18(weights=weights)
        self.feature_dim = resnet.fc.in_features          # 512
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        return torch.flatten(self.backbone(x), 1)

    def train(self, mode=True):
        """
        Put the network in train mode but keep every BatchNorm2d in eval mode
        so that running statistics (mean/var) are never updated during fine-tuning.
        gamma (weight) and beta (bias) stay trainable.
        """
        super().train(mode)
        if mode:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if m.weight is not None:
                        m.weight.requires_grad_(True)
                    if m.bias is not None:
                        m.bias.requires_grad_(True)


class ClassifierHead(nn.Module):
    def __init__(self, in_features=512, num_classes=7):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)
