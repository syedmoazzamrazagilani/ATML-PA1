import torch.nn as nn


class ERMLoss(nn.Module):
    """
    Plain cross-entropy classification loss.
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        return self.criterion(logits, labels)
