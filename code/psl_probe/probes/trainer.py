"""Probe fitting loop.

Probes are small enough that the entire feature matrix fits in GPU memory, so we
train with full-precision AdamW and early stopping on validation macro-MCC. The
loop is intentionally simple: any complexity added here would be capacity handed
to the probe, which would blur the representation-analysis conclusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from psl_probe.analysis.metrics import multilabel_metrics, sigmoid, tune_thresholds
from psl_probe.config import ProbeConfig
from psl_probe.probes.linear import (
    FeatureStandardizer,
    build_probe,
    positive_class_weights,
    predict_logits,
)
from psl_probe.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ProbeResult:
    """Everything needed to report and reproduce one probe fit."""

    metrics: dict[str, float]
    per_class: list[dict]
    thresholds: np.ndarray
    best_epoch: int
    n_train: int
    n_val: int
    n_test: int
    history: list[dict] = field(default_factory=list)
    state_dict: dict | None = None
    #: Test-split predicted probabilities and labels, ``(n_test, n_classes)`` each.
    #: Retained so that comparisons between two probes (e.g. baseline vs ablated)
    #: can be bootstrapped *paired* on the same proteins without refitting. At a
    #: few hundred kilobytes per fit this costs nothing worth optimising away.
    test_scores: np.ndarray | None = None
    test_labels: np.ndarray | None = None

    def to_record(self, **extra) -> dict:
        """Flatten into a JSON-serialisable row for the results table."""
        record = {**extra, **self.metrics}
        record.update(
            {
                "best_epoch": self.best_epoch,
                "n_train": self.n_train,
                "n_val": self.n_val,
                "n_test": self.n_test,
            }
        )
        return record


def train_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
    cfg: ProbeConfig,
    device: str = "cuda",
    seed: int = 0,
    keep_state: bool = False,
) -> ProbeResult:
    """Fit a probe and evaluate it on the held-out test split.

    Decision thresholds are tuned on the validation split and then applied
    unchanged to the test split, so no test information influences any reported
    number.

    Parameters
    ----------
    x_train, x_val, x_test:
        Feature matrices of shape ``(n, hidden)``.
    y_train, y_val, y_test:
        Multi-hot label matrices of shape ``(n, n_classes)``.
    class_names:
        Names in label-column order, used for per-class reporting.
    cfg:
        Probe hyper-parameters.
    device:
        Torch device string.
    seed:
        Seed for probe initialisation and shuffling.
    keep_state:
        Retain the trained weights on the result (needed for weight-space analyses;
        omitted by default because a full layer sweep would otherwise hold hundreds
        of parameter tensors in memory).
    """
    torch.manual_seed(seed)
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")

    to_tensor = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=torch_device)  # noqa: E731
    xtr, ytr = to_tensor(x_train), to_tensor(y_train)
    xva, yva = to_tensor(x_val), to_tensor(y_val)
    xte, yte = to_tensor(x_test), to_tensor(y_test)

    # Reject non-finite features up front. Otherwise they propagate silently
    # through training and only surface as an opaque "Input contains NaN" from
    # inside a sklearn metric, several frames from the real cause.
    for name, features in (("train", xtr), ("val", xva), ("test", xte)):
        if not torch.isfinite(features).all():
            n_bad = int((~torch.isfinite(features)).sum().item())
            n_rows = int((~torch.isfinite(features)).any(dim=1).sum().item())
            raise FloatingPointError(
                f"{name} features contain {n_bad} non-finite values across "
                f"{n_rows} proteins. The embedding store is corrupt -- most likely "
                f"written before the fp16 pooling-overflow fix. Delete it and "
                f"re-run psl_probe.cli.extract_embeddings."
            )

    scaler: FeatureStandardizer | None = None
    if cfg.standardize:
        scaler = FeatureStandardizer().fit(xtr)
        xtr, xva, xte = scaler.transform(xtr), scaler.transform(xva), scaler.transform(xte)

    model = build_probe(cfg, input_dim=xtr.shape[1], n_classes=len(class_names)).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pos_weight = positive_class_weights(ytr, cfg.class_weighting)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_score = -np.inf
    best_state: dict[str, Tensor] = {}
    best_epoch = -1
    epochs_without_improvement = 0
    history: list[dict] = []

    generator = torch.Generator(device="cpu").manual_seed(seed)
    n_train = xtr.shape[0]

    for epoch in range(cfg.epochs):
        model.train()
        permutation = torch.randperm(n_train, generator=generator).to(torch_device)
        epoch_loss = 0.0
        for start in range(0, n_train, cfg.batch_size):
            index = permutation[start : start + cfg.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xtr[index]), ytr[index])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(index)

        val_scores = sigmoid(predict_logits(model, xva).cpu().numpy())
        val_metrics = multilabel_metrics(yva.cpu().numpy(), val_scores, class_names)
        history.append(
            {"epoch": epoch, "train_loss": epoch_loss / n_train, **{f"val_{k}": v for k, v in val_metrics.items()}}
        )

        if val_metrics["macro_mcc"] > best_score:
            best_score = val_metrics["macro_mcc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.early_stopping_patience:
                logger.debug("Early stopping at epoch %d (best epoch %d)", epoch, best_epoch)
                break

    model.load_state_dict(best_state)

    val_scores = sigmoid(predict_logits(model, xva).cpu().numpy())
    thresholds = tune_thresholds(yva.cpu().numpy(), val_scores)

    test_scores = sigmoid(predict_logits(model, xte).cpu().numpy())
    y_test_np = yte.cpu().numpy()
    from psl_probe.analysis.metrics import per_class_metrics

    result = ProbeResult(
        metrics=multilabel_metrics(y_test_np, test_scores, class_names, thresholds),
        per_class=per_class_metrics(y_test_np, test_scores, class_names, thresholds),
        thresholds=thresholds,
        best_epoch=best_epoch,
        n_train=int(xtr.shape[0]),
        n_val=int(xva.shape[0]),
        n_test=int(xte.shape[0]),
        history=history,
        test_scores=test_scores,
        test_labels=y_test_np,
    )
    if keep_state:
        state = {k: v.cpu() for k, v in best_state.items()}
        if scaler is not None:
            state["_scaler"] = scaler.state_dict()  # type: ignore[assignment]
        result.state_dict = state
    return result


def save_probe(result: ProbeResult, path: Path | str) -> Path:
    """Persist probe weights and thresholds for later reuse."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": result.state_dict, "thresholds": result.thresholds,
         "metrics": result.metrics, "best_epoch": result.best_epoch},
        path,
    )
    return path
