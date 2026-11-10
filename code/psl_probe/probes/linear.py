"""Probe architectures.

Probes are deliberately low-capacity. A linear probe measures how *linearly
decodable* localization is from a frozen representation, which is the quantity of
interest for representation analysis; the one-hidden-layer MLP is reported as a
capacity control, to distinguish "the information is absent" from "the information
is present but not linearly separable".
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from psl_probe.config import ProbeConfig


class LinearProbe(nn.Module):
    """Multi-label linear probe: a single affine map to per-class logits."""

    def __init__(self, input_dim: int, n_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, n_classes)
        nn.init.zeros_(self.linear.bias)

    def forward(self, features: Tensor) -> Tensor:
        """Return logits of shape ``(batch, n_classes)``."""
        return self.linear(features)


class MLPProbe(nn.Module):
    """Capacity control: one hidden layer with GELU and dropout."""

    def __init__(
        self, input_dim: int, n_classes: int, hidden_dim: int = 256, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


def build_probe(cfg: ProbeConfig, input_dim: int, n_classes: int) -> nn.Module:
    """Instantiate the probe named by ``cfg.kind``."""
    if cfg.kind == "linear":
        return LinearProbe(input_dim, n_classes)
    if cfg.kind == "mlp":
        return MLPProbe(input_dim, n_classes, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout)
    raise ValueError(f"Unknown probe kind: {cfg.kind!r}")


def positive_class_weights(labels: Tensor, mode: str = "balanced") -> Tensor | None:
    """Per-class positive weights for :class:`~torch.nn.BCEWithLogitsLoss`.

    With ``mode="balanced"`` the weight for class ``c`` is ``n_negative / n_positive``,
    which equalises the loss contribution of the two classes. This matters here
    because compartment prevalence spans roughly two orders of magnitude, and an
    unweighted probe would simply never predict the rare compartments.
    """
    if mode == "none":
        return None
    if mode != "balanced":
        raise ValueError(f"Unknown class_weighting: {mode!r}")
    n_positive = labels.sum(dim=0)
    n_negative = labels.shape[0] - n_positive
    # Clamp guards against a class that is absent from a fold.
    return (n_negative / n_positive.clamp(min=1.0)).clamp(max=100.0)


class FeatureStandardizer:
    """Zero-mean/unit-variance scaling fitted on the training split only.

    Fitting the scaler on training data alone is essential: leaking test-split
    statistics inflates probe accuracy, particularly for the rare compartments.
    """

    def __init__(self) -> None:
        self.mean: Tensor | None = None
        self.std: Tensor | None = None

    def fit(self, features: Tensor) -> FeatureStandardizer:
        self.mean = features.mean(dim=0, keepdim=True)
        self.std = features.std(dim=0, keepdim=True).clamp(min=1e-6)
        return self

    def transform(self, features: Tensor) -> Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("FeatureStandardizer must be fitted before transform().")
        return (features - self.mean) / self.std

    def fit_transform(self, features: Tensor) -> Tensor:
        return self.fit(features).transform(features)

    def state_dict(self) -> dict[str, Tensor]:
        if self.mean is None or self.std is None:
            raise RuntimeError("FeatureStandardizer is not fitted.")
        return {"mean": self.mean.cpu(), "std": self.std.cpu()}

    def load_state_dict(self, state: dict[str, Tensor]) -> FeatureStandardizer:
        self.mean = state["mean"]
        self.std = state["std"]
        return self


@torch.no_grad()
def predict_logits(model: nn.Module, features: Tensor, batch_size: int = 4096) -> Tensor:
    """Batched inference that keeps peak memory bounded for large splits."""
    model.eval()
    chunks = [model(features[i : i + batch_size]) for i in range(0, len(features), batch_size)]
    return torch.cat(chunks, dim=0)
