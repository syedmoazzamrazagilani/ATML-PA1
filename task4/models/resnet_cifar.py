import torch
import torch.nn as nn
import torchvision.models as models


class CifarResNet18(nn.Module):
    """
    Standard 10-class ResNet-18 for CIFAR (32×32).
    - 3×3 stride-1 first conv, no initial max-pool.
    - Final FC outputs `num_classes` logits.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        resnet = models.resnet18(weights=None)

        # Replace first conv and remove max-pool
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                  padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

        self.model      = resnet
        self.feature_dim = 512     # dimension before FC

    def forward(self, x):
        return self.model(x)

    def extract_features(self, x):
        """Returns the penultimate (512-d) feature vector."""
        m = self.model
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)     # Identity
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)
        x = m.avgpool(x)
        return torch.flatten(x, 1)

    def logits_from_features(self, feat):
        return self.model.fc(feat)


class ProserResNet18(nn.Module):
    """
    ResNet-18 for PROSER.

    Adds `num_dummy_classes` placeholder classifiers on top of the
    `num_known_classes` real classifiers.

    forward(x) → (logits_all, feat)
      logits_all : (B, num_known + num_dummy)
      feat        : (B, 512)  penultimate feature

    forward_pre(x)  → feature after layer2  (for manifold mixup)
    forward_post(h) → (logits_all, feat)    from layer2 feature onward
    """

    def __init__(self, num_known_classes=10, num_dummy_classes=5):
        super().__init__()
        resnet = models.resnet18(weights=None)

        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                                  padding=1, bias=False)
        resnet.maxpool = nn.Identity()

        self.conv1   = resnet.conv1
        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4
        self.avgpool = resnet.avgpool

        self.feature_dim       = 512
        self.num_known         = num_known_classes
        self.num_dummy         = num_dummy_classes
        self.num_total_classes = num_known_classes + num_dummy_classes

        self.fc = nn.Linear(512, self.num_total_classes)

    def _stem(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

    def forward_pre(self, x):
        """Returns intermediate feature after layer2 (for manifold mixup)."""
        x = self._stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x

    def forward_post(self, h):
        """Completes the forward pass from the layer2 feature."""
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.avgpool(h)
        feat    = torch.flatten(h, 1)
        logits  = self.fc(feat)
        return logits, feat

    def forward(self, x):
        h       = self.forward_pre(x)
        logits, feat = self.forward_post(h)
        return logits, feat

    def known_logits(self, x):
        """Returns only the `num_known` logits (for CSA / MLS evaluation)."""
        logits, feat = self.forward(x)
        return logits[:, :self.num_known], feat

    @classmethod
    def from_vanilla_ckpt(cls, ckpt_path, num_known=10, num_dummy=5, device="cpu"):
        """
        Initialise from a saved CifarResNet18 checkpoint.
        Known-class FC weights are copied; dummy classifiers are random.
        """
        model = cls(num_known_classes=num_known, num_dummy_classes=num_dummy)
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state", ckpt)  # handle both formats

        model_state = model.state_dict()
        for k, v in state.items():
            if k == "model.fc.weight":
                model_state["fc.weight"][:num_known] = v
            elif k == "model.fc.bias":
                model_state["fc.bias"][:num_known] = v
            else:
                # Remap "model.layerX" → "layerX" etc.
                new_k = k.replace("model.", "", 1)
                if new_k in model_state:
                    model_state[new_k] = v
        model.load_state_dict(model_state, strict=False)
        return model.to(device)
