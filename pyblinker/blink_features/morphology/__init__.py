"""Morphology feature module."""

from .core_metrics import compute_blink_morphology_metrics

__all__ = [
    "compute_blink_morphology_metrics",
    "MorphologyBlinkFeatureExtractor",
    "compute_epoch_morphology_features",
    "compute_morphology_features",
    "compute_blink_waveform_metrics",
]


def __getattr__(name: str):
    if name == "compute_epoch_morphology_features":
        from .epoch_features import compute_epoch_morphology_features

        return compute_epoch_morphology_features
    if name == "compute_morphology_features":
        from .epoch_features import compute_morphology_features

        return compute_morphology_features
    if name == "MorphologyBlinkFeatureExtractor":
        from .epoch_features import MorphologyBlinkFeatureExtractor

        return MorphologyBlinkFeatureExtractor
    if name == "compute_blink_waveform_metrics":
        from .per_blink import compute_blink_waveform_metrics

        return compute_blink_waveform_metrics
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
