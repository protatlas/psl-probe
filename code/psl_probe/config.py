"""Typed configuration objects.

Configuration is expressed as nested dataclasses that are hydrated from YAML
(``configs/*.yaml``). Every CLI accepts ``--config`` plus dotted
``--set key.subkey=value`` overrides, so the exact resolved configuration can be
serialised next to each result for reproducibility.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from psl_probe.utils.io import load_yaml

# Repository root, resolved relative to this file (code/psl_probe/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PathConfig:
    """Filesystem layout. Relative paths are resolved against the repository root."""

    raw: Path = Path("datasets/raw")
    interim: Path = Path("datasets/interim")
    processed: Path = Path("datasets/processed")
    embeddings: Path = Path("results/embeddings")
    probes: Path = Path("results/probes")
    analysis: Path = Path("results/analysis")
    logs: Path = Path("results/logs")
    figures: Path = Path("visualization/figures")
    #: LaTeX tables are generated, never hand-edited; configurable so a smoke or
    #: exploratory run cannot overwrite the manuscript's tables.
    tables: Path = Path("latex/tables")

    def __post_init__(self) -> None:
        for f in dataclasses.fields(self):
            value = Path(getattr(self, f.name))
            if not value.is_absolute():
                value = REPO_ROOT / value
            setattr(self, f.name, value)

    def mkdirs(self) -> None:
        """Create every configured directory."""
        for f in dataclasses.fields(self):
            Path(getattr(self, f.name)).mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Benchmark construction parameters."""

    #: Swiss-Prot release used; ``current_release`` tracks the live UniProt FTP tree.
    uniprot_release: str = "current_release"
    #: Restrict to these NCBI superkingdoms (empty list = no filter).
    kingdoms: list[str] = field(default_factory=lambda: ["Eukaryota"])
    min_length: int = 40
    max_length: int = 4096
    #: Keep only subcellular-location annotations with experimental evidence codes.
    require_experimental_evidence: bool = True
    #: Drop proteins whose annotations map to more than this many compartments.
    max_labels_per_protein: int = 3
    #: MMseqs2 sequence-identity threshold for homology partitioning.
    homology_identity: float = 0.30
    homology_coverage: float = 0.80
    n_folds: int = 5
    test_fraction: float = 0.20


@dataclass
class ModelConfig:
    """Protein language model to probe."""

    #: Hugging Face model id, e.g. ``facebook/esm2_t33_650M_UR50D``.
    hf_id: str = "facebook/esm2_t33_650M_UR50D"
    #: Short tag used in filenames and figure legends.
    tag: str = "esm2_650m"
    family: str = "esm2"  # one of {"esm2", "prott5", "ankh"}
    n_layers: int = 33
    hidden_size: int = 1280
    #: Layers to extract; ``null`` in YAML means "all layers including embeddings".
    layers: list[int] | None = None
    max_length: int = 1024
    batch_tokens: int = 16384
    dtype: str = "float16"
    output_attentions: bool = False


@dataclass
class PoolingConfig:
    """How per-residue representations are reduced to fixed-size vectors."""

    #: Any subset of {"mean", "cls", "nterm", "cterm", "max"}.
    strategies: list[str] = field(default_factory=lambda: ["mean", "nterm"])
    #: Window length for the N-terminal pool (covers most signal/transit peptides).
    nterm_window: int = 70
    cterm_window: int = 30
    #: Store per-residue tensors as well (large; needed for attention analyses).
    keep_per_residue: bool = False


@dataclass
class ProbeConfig:
    """Probe architecture and optimisation."""

    #: "linear" (single affine layer) or "mlp" (one hidden layer).
    kind: str = "linear"
    hidden_dim: int = 256
    dropout: float = 0.1
    weight_decay: float = 1e-4
    lr: float = 1e-3
    epochs: int = 60
    batch_size: int = 256
    early_stopping_patience: int = 10
    #: Standardise features with train-split statistics before fitting.
    standardize: bool = True
    #: Fit matched random-label control tasks for selectivity (Hewitt & Liang, 2019).
    control_task: bool = True
    control_seeds: int = 3
    #: Positive-class weighting for the multi-label BCE objective.
    class_weighting: str = "balanced"  # {"none", "balanced"}


@dataclass
class AnalysisConfig:
    """Downstream representation analyses."""

    run_cka: bool = True
    cka_max_proteins: int = 4000
    run_attention: bool = True
    #: Attention analysis is restricted to proteins with a curated SIGNAL/TRANSIT feature.
    attention_max_proteins: int = 2000
    run_ablation: bool = True
    #: N-terminal prefix lengths to mask in the sequence-ablation experiment.
    ablation_prefixes: list[int] = field(default_factory=lambda: [0, 10, 20, 40, 70])
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95


@dataclass
class Config:
    """Root configuration object."""

    seed: int = 20260905
    device: str = "cuda"
    num_workers: int = 8
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pooling: PoolingConfig = field(default_factory=PoolingConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: Path | str) -> Config:
        """Build a config from a YAML file, supporting a single ``base:`` include."""
        path = Path(path)
        payload = load_yaml(path)

        base_name = payload.pop("base", None)
        if base_name is not None:
            base_payload = load_yaml(path.parent / base_name)
            payload = _deep_merge(base_payload, payload)

        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Config:
        """Recursively hydrate nested dataclasses from a plain dictionary."""
        return _from_dict(cls, payload)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain Python types (``Path`` rendered as ``str``)."""
        return _to_plain(dataclasses.asdict(self))

    def apply_overrides(self, overrides: list[str] | None) -> Config:
        """Apply ``dotted.key=value`` overrides parsed as YAML scalars."""
        import yaml

        for override in overrides or []:
            if "=" not in override:
                raise ValueError(f"Override must be of the form key=value, got {override!r}")
            dotted, raw_value = override.split("=", 1)
            value = yaml.safe_load(raw_value)
            target: Any = self
            parts = dotted.split(".")
            for part in parts[:-1]:
                target = getattr(target, part)
            if not hasattr(target, parts[-1]):
                raise AttributeError(f"Unknown config key: {dotted}")
            setattr(target, parts[-1], value)
        # Re-resolve derived state (e.g. relative paths turned absolute).
        self.paths.__post_init__()
        return self


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _from_dict(cls: type, payload: dict[str, Any]) -> Any:
    """Instantiate a (possibly nested) dataclass from a dictionary."""
    kwargs: dict[str, Any] = {}
    known = {f.name for f in dataclasses.fields(cls)}
    for key, value in payload.items():
        if key not in known:
            raise KeyError(f"Unknown configuration key {key!r} for {cls.__name__}")
        # Nested dataclass fields arrive as dictionaries.
        nested = _resolve_dataclass(cls, key)
        if nested is not None and isinstance(value, dict):
            kwargs[key] = _from_dict(nested, value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _resolve_dataclass(cls: type, field_name: str) -> type | None:
    """Return the dataclass type of ``field_name`` on ``cls``, if it is one."""
    for f in dataclasses.fields(cls):
        if f.name != field_name:
            continue
        # ``default_factory`` is the reliable handle on nested dataclass types,
        # because ``f.type`` may be a postponed (string) annotation.
        if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            candidate = f.default_factory  # type: ignore[misc]
            if dataclasses.is_dataclass(candidate):
                return candidate  # type: ignore[return-value]
        if dataclasses.is_dataclass(f.type):
            return f.type  # type: ignore[return-value]
    return None


def _to_plain(obj: Any) -> Any:
    """Convert ``Path`` objects to strings recursively for serialisation."""
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj
