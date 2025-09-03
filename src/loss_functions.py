import torch

import torch.nn as nn
import torch.nn.functional as F

def _normalize_alpha(alpha: torch.Tensor, mode: str | None = "max1") -> torch.Tensor:
    if mode is None:
        return alpha
    if mode == "mean1":  # average weight = 1
        return alpha * (alpha.numel() / alpha.sum())
    if mode == "max1":   # strongest class = 1 (keeps focal <= CE if gamma>=0)
        return alpha / alpha.max()
    raise ValueError(mode)


def alpha_from_counts(
    counts: torch.Tensor,
    scheme: str = "effective",  # ["inverse","effective","median","sqrt_inv"]
    beta: float = 0.999,        # only used by "effective"
    normalize: str | None = "max1",
) -> torch.Tensor:
    counts = counts.to(torch.float32).clamp_min_(1)
    if scheme == "inverse":
        alpha = 1.0 / counts
    elif scheme == "effective":
        # Class-Balanced (Cui et al. CVPR'19): (1 - beta) / (1 - beta^n_c)
        eff = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float32, device=counts.device), counts)
        alpha = (1.0 - beta) / eff
    elif scheme == "median":
        alpha = torch.median(counts) / counts
    elif scheme == "sqrt_inv":
        alpha = 1.0 / torch.sqrt(counts)
    else:
        raise ValueError(scheme)
    # first bring mean near 1 (nice for comparing schemes)
    alpha = alpha / alpha.mean()
    # then apply chosen normalization
    return _normalize_alpha(alpha, normalize)



# ---------- shared helpers ----------
def class_counts_from_tensor(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.bincount(labels, minlength=num_classes).clamp_min(1)

def class_balanced_weights_from_counts(counts: torch.Tensor, beta: float = 0.999, normalize: bool = True) -> torch.Tensor:
    # Effective number: (1 - beta^n_c)
    eff_num = 1.0 - torch.pow(torch.tensor(beta, device=counts.device, dtype=torch.float32), counts.to(torch.float32))
    w = (1.0 - beta) / eff_num
    if normalize:
        w = w * (counts.numel() / w.sum())
    return w

# ---------- Loss 1: Class-Balanced (Effective Number) + CE ----------
class ClassBalancedCELoss(nn.Module):
    def __init__(self, counts: torch.Tensor, beta: float = 0.999, reduction: str = "mean"):
        super().__init__()
        self.register_buffer("counts", counts.to(torch.long))
        self.beta = beta
        self.reduction = reduction
        self._refresh()

    def _refresh(self):
        self.register_buffer("weights", class_balanced_weights_from_counts(self.counts, beta=self.beta, normalize=True))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.weights, reduction=self.reduction)

# ---------- Loss 2: Focal Loss (multiclass; α can be scalar or vector) ----------
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        if alpha is None:
            self.alpha = None
        else:
            self.alpha = torch.as_tensor(alpha, dtype=torch.float32)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction='none')     # -log p_t
        pt = torch.exp(-ce).clamp(min=1e-12, max=1-1e-12)          # p_t

        if self.alpha is None:
            alpha_t = torch.ones_like(ce)
        elif self.alpha.ndim == 0:
            alpha_t = torch.full_like(ce, float(self.alpha.item()))
        else:
            alpha_t = self.alpha.to(logits.device)[targets]

        loss = alpha_t * (1.0 - pt).pow(self.gamma) * ce
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum":  return loss.sum()
        return loss

# ---------- Loss 3: LDAM + DRW ----------
class LDAMDRWLoss(nn.Module):
    """
    LDAM margin with optional DRW schedule:
      - subtract m_c from the target logit (m_c = C_margin / n_c^(1/4))
      - before drw_start: plain LDAM-CE
      - after  drw_start: LDAM-CE with class-balanced weights (effective number)
    Call `set_epoch(e)` once per epoch.
    """
    def __init__(self, counts: torch.Tensor, C_margin: float = 0.5, drw_start: int = 10, cb_beta: float = 0.999, reduction: str = "mean"):
        super().__init__()
        counts = counts.to(torch.long)
        self.register_buffer("counts", counts)
        self.register_buffer("m_c", (C_margin / (counts.to(torch.float32).pow(0.25))))
        self.drw_start = drw_start
        self.cb_beta = cb_beta
        self.reduction = reduction
        self.epoch = 0
        # warmup weights (None)
        self.register_buffer("weights", None)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        if self.epoch >= self.drw_start:
            w = class_balanced_weights_from_counts(self.counts, beta=self.cb_beta, normalize=True)
            self.register_buffer("weights", w)
        else:
            self.register_buffer("weights", None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # subtract margin on target logit
        idx = torch.arange(logits.size(0), device=logits.device)
        logits_adj = logits.clone()
        logits_adj[idx, targets] = logits_adj[idx, targets] - self.m_c[targets].to(logits.device)

        return F.cross_entropy(logits_adj, targets, weight=self.weights, reduction=self.reduction)

# ---------- Loss 4: Logit-Adjusted CE (priors) ----------
class LogitAdjustedCELoss(nn.Module):
    def __init__(self, counts: torch.Tensor, tau: float = 1.0, reduction: str = "mean"):
        super().__init__()
        priors = (counts.to(torch.float32) / counts.sum()).clamp_(min=1e-12)
        self.register_buffer("adj", tau * priors.log())
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits + self.adj.to(logits.device), targets, reduction=self.reduction)

# ---------- Loss 5: Balanced Softmax ----------
class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, counts: torch.Tensor, reduction: str = "mean"):
        super().__init__()
        self.register_buffer("adj", counts.to(torch.float32).clamp_min(1).log())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits + self.adj.to(logits.device), targets, reduction="mean")

# ---------- Loss factory ----------
def build_imbalance_loss(loss_name: str,
                         num_classes: int,
                         counts: torch.Tensor | None = None,
                         focal_gamma: float = 2.0,
                         focal_alpha=None,
                         cb_beta: float = 0.999,
                         ldam_C_margin: float = 0.5,
                         drw_start: int = 10,
                         cb_beta_drw: float = 0.999,
                         logit_adj_tau: float = 1.0):
    """
    Returns (criterion, needs_epoch_hook)
    """
    needs_counts = loss_name in {"ce_cb","ldam_drw","logit_adj","balanced_softmax"}
    if needs_counts and counts is None:
        raise ValueError(f"{loss_name} requires class counts (tensor shape [C]).")

    if loss_name == "vanilla_ce":
        return nn.CrossEntropyLoss(), False
    if loss_name == "ce_cb":
        return ClassBalancedCELoss(counts=counts, beta=cb_beta), False
    if loss_name == "focal":
        return FocalLoss(gamma=focal_gamma, alpha=focal_alpha), False
    if loss_name == "ldam_drw":
        return LDAMDRWLoss(counts=counts, C_margin=ldam_C_margin, drw_start=drw_start, cb_beta=cb_beta_drw), True
    if loss_name == "logit_adj":
        return LogitAdjustedCELoss(counts=counts, tau=logit_adj_tau), False
    if loss_name == "balanced_softmax":
        return BalancedSoftmaxLoss(counts=counts), False
    raise ValueError(f"Unknown loss_name: {loss_name}")
