"""Benchmark construction from Swiss-Prot."""

from psl_probe.data.labels import (
    LOCALIZATION_CLASSES,
    SIGNAL_FEATURE_TYPES,
    labels_to_vector,
    map_subcellular_location,
)

__all__ = [
    "LOCALIZATION_CLASSES",
    "SIGNAL_FEATURE_TYPES",
    "labels_to_vector",
    "map_subcellular_location",
]
