"""Blink kinematic feature package."""

from .core_metrics import compute_blink_kinematic_metrics

__all__ = [
    "compute_blink_kinematic_metrics",
    "compute_kinematic_features",
    "KinematicBlinkFeatureExtractor",
]


def __getattr__(name: str):
    if name == "compute_kinematic_features":
        from .kinematic_features import compute_kinematic_features

        return compute_kinematic_features
    if name == "KinematicBlinkFeatureExtractor":
        from .kinematic_features import KinematicBlinkFeatureExtractor

        return KinematicBlinkFeatureExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
