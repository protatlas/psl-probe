"""Configuration hydration, overrides and provenance round-tripping."""

from __future__ import annotations

import pytest

from psl_probe.config import REPO_ROOT, Config


def test_defaults_resolve_paths_against_repo_root() -> None:
    cfg = Config()
    assert cfg.paths.processed.is_absolute()
    assert cfg.paths.processed == REPO_ROOT / "datasets" / "processed"


def test_yaml_include_merges_base() -> None:
    cfg = Config.from_yaml(REPO_ROOT / "configs" / "esm2_650m.yaml")
    assert cfg.model.tag == "esm2_650m"       # from the model file
    assert cfg.data.n_folds == 5              # inherited from base.yaml


def test_dotted_overrides_apply() -> None:
    cfg = Config().apply_overrides(["probe.epochs=3", "model.tag=tiny", "data.kingdoms=[]"])
    assert cfg.probe.epochs == 3
    assert cfg.model.tag == "tiny"
    assert cfg.data.kingdoms == []


def test_unknown_override_key_raises() -> None:
    with pytest.raises(AttributeError):
        Config().apply_overrides(["probe.nonexistent=1"])


def test_malformed_override_raises() -> None:
    with pytest.raises(ValueError, match="key=value"):
        Config().apply_overrides(["probe.epochs"])


def test_roundtrip_through_dict_is_lossless() -> None:
    cfg = Config().apply_overrides(["probe.kind=mlp", "seed=42"])
    restored = Config.from_dict(cfg.to_dict())
    assert restored.probe.kind == "mlp"
    assert restored.seed == 42
    assert restored.paths.processed == cfg.paths.processed
