import torch
import torch.nn as nn


class SAMOptimizer:
    """
    Wraps an existing AdamW optimizer to implement the SAM two-step update.
    Usage:
        sam = SAMOptimizer(base_optimizer, rho=0.05)

        # --- step 1: ascent ---
        loss = criterion(model(x), y)
        loss.backward()
        sam.first_step(zero_grad=True)   # perturbs params, zeros grad

        # --- step 2: descent ---
        loss2 = criterion(model(x), y)   # forward at perturbed point
        loss2.backward()
        sam.second_step(zero_grad=True)  # updates original params
    """
    def __init__(self, base_optimizer: torch.optim.Optimizer, rho: float = 0.05):
        self.base_optimizer = base_optimizer
        self.rho            = rho
        self._perturbations  = {}        
      
    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """
        Compute normalised gradient ascent step and add perturbation to params.
        Saves the perturbation so second_step can subtract it.
        """
        grad_norm = self._grad_norm()
        if grad_norm == 0.0:
            return

        scale = self.rho / (grad_norm + 1e-12)
        self._perturbations = {}

        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                eps = p.grad * scale
                p.add_(eps)
                self._perturbations[id(p)] = eps

        if zero_grad:
            self.base_optimizer.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """
        Subtract the perturbation (restore original params), then run
        the base AdamW step using the gradient at the perturbed point.
        """
        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                if id(p) in self._perturbations:
                    p.sub_(self._perturbations[id(p)])

        self.base_optimizer.step()
        if zero_grad:
            self.base_optimizer.zero_grad()

        self._perturbations = {}

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, sd):
        self.base_optimizer.load_state_dict(sd)

    def _grad_norm(self) -> float:
        norms = []
        for group in self.base_optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    norms.append(p.grad.detach().norm(2))
        if not norms:
            return 0.0
        return torch.stack(norms).norm(2).item()


class ERMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        return self.criterion(logits, labels)
