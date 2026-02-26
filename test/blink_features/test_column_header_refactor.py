"""Schema snapshot tests for column header modules."""

from pyblinker.blink_features.energy.column_headers import build_output_columns as build_energy_columns
from pyblinker.blink_features.kinematics.column_headers import (
    EXTENDED_METRICS,
    STATS,
    build_output_columns as build_kinematic_columns,
    metrics_for_style as kinematic_metrics_for_style,
)
from pyblinker.blink_features.morphology.column_headers import (
    LEGACY_MORPHOLOGY_METRICS,
    STATS as MORPH_STATS,
    build_output_columns as build_morphology_columns,
    metrics_for_style as morphology_metrics_for_style,
    rename_metric_column_name,
)


def test_energy_column_order_matches_legacy_logic() -> None:
    modality_by_channel = {"EEG-E8": "eeg", "EOG-EEG-eog_vert_left": "eog"}
    styles_by_modality = {"eeg": {"base", "peak"}, "eog": {"base"}}

    new_columns = build_energy_columns(modality_by_channel, styles_by_modality)

    expected = []
    metrics = [
        "blink_signal_energy",
        "teager_kaiser_energy",
        "blink_line_length",
        "blink_velocity_integral",
    ]
    stats = ["mean", "std", "cv"]
    for ch, modality in modality_by_channel.items():
        for style in sorted(styles_by_modality.get(modality, set())):
            for metric in metrics:
                for stat in stats:
                    channel = ch if modality == "eog" else ch.upper()
                    expected.append(f"{modality}__{style}__energy__{metric}_{stat}__{channel}")

    assert new_columns == expected


def test_kinematic_column_schema_matches_legacy_set_logic() -> None:
    modality_channels = {"eeg": ["EEG-E8"], "ear": ["EAR-avg_ear"]}
    styles_by_modality = {"eeg": {"base", "tent"}, "ear": {"th_point"}}

    new_columns = build_kinematic_columns(modality_channels, styles_by_modality)

    expected_set = set()
    for mod, channels in modality_channels.items():
        for style in sorted(styles_by_modality.get(mod) or {"base"}):
            metrics = kinematic_metrics_for_style(style)
            metrics.extend(EXTENDED_METRICS)
            for metric in metrics:
                for stat in STATS:
                    for ch in channels:
                        expected_set.add(f"{mod}__{style}__kinematic__{metric}_{stat}__{ch}")

    assert new_columns == sorted(expected_set)


def test_morphology_schema_matches_legacy_set_plus_legacy_columns() -> None:
    modality_channels = {"eeg": ["EEG-E8"], "ear": ["EAR-avg_ear"]}
    styles_by_modality = {"eeg": {"base", "peak"}, "ear": {"th_point"}}

    new_columns = build_morphology_columns(modality_channels, styles_by_modality)

    expected_set = set()
    for mod, channels in modality_channels.items():
        for style in sorted(styles_by_modality.get(mod) or {"base"}):
            for metric in morphology_metrics_for_style(style):
                for stat in MORPH_STATS:
                    for ch in channels:
                        expected_set.add(f"{mod}__{style}__morphology__{metric}_{stat}__{ch}")

        if channels and mod in {"eeg", "eog"}:
            primary_channel = channels[0]
            for legacy_metric in LEGACY_MORPHOLOGY_METRICS:
                for stat_name in MORPH_STATS:
                    expected_set.add(
                        rename_metric_column_name(
                            modality=mod,
                            metric=legacy_metric,
                            stat_name=stat_name,
                            channel_name=primary_channel,
                        )
                    )

    assert new_columns == sorted(expected_set)
