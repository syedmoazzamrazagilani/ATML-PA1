import torch.nn as nn

class DANNLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, domain_logits, domain_labels):
        return self.criterion(domain_logits, domain_labels)
