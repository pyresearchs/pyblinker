Refactor roadmap
Phase 1 – safe refactors (no behaviour change)
1.	Centralise shared constants – Create a module blink_features/constants.py containing _STATS, the metric registry per family, and modality inference. Replace all duplicated definitions with imports from this module. Add a configuration dataclass (e.g., BlinkerConfig) encapsulating thresholds like shut_amp_fraction, base_fraction, p_avr_threshold and z_thresholds. Provide default values via default_setting.py but allow callers to override via parameters or environment variables.
2.	To delete, no excuse
To delete the  as we are not going to use approach of ONSET and duration prefix >> utils.style_windows.available_styles(metadata_columns, modality, *, onset_prefix='onset__', duration_prefix='duration__')
3.	Extract style/window helpers – Implement utils.style_windows.available_styles(metadata_columns, modality, *, onset_prefix='onset__', duration_prefix='duration__') and style_windows.extract_windows(metadata_row, modality, style, n_times, *, start_prefix='start__', end_prefix='end__'). Parameterising prefixes allows reuse across families.
4.	Modularise kinematic feature code – Move low‑level functions such as _coerce_numeric_list, _pad, _initialize_extended_columns and metric computations into a new kinematics/helpers.py module. Mark internal helpers with a leading underscore and document them.
5.	Mark optional dependencies – Wrap imports of external packages (e.g., pywt) in try/except blocks and provide informative error messages or fallbacks so that tests that do not need these features can still run. Consider using Python’s importlib.resources to provide vendorised wavelet filters.
Phase 2 – consolidation (shared loop skeletons & config migration)
1.	Design a common compute skeleton – Create an internal function compute_features(epochs, picks, metrics_by_style, compute_func, config, *, family_name) that encapsulates the orchestration pattern: resolve channels, infer modality, determine styles, build column names, iterate over epochs and channels, call compute_func for the actual metric computations, and assemble the DataFrame. Each feature family would supply its own metrics_by_style (mapping from style to metric list) and a callback compute_func(style, signal_segment, sfreq, **kwargs) returning per‑metric statistics. This would drastically reduce duplication.
2.	Refactor existing extractors – Modify MorphologyBlinkFeatureExtractor.compute(), KinematicBlinkFeatureExtractor.compute() energy-domain, and the frequency‑domain aggregate function to delegate to the shared compute skeleton. Feature‑specific logic remains in their callback functions (e.g., computing waveform metrics, kinematic metrics or wavelet energies). Remove now redundant helper functions and constants.
3.	Migrate configuration – Replace scattered constants with values from BlinkerConfig. Expose configuration parameters to callers via high‑level APIs (e.g., compute_epoch_morphology_features(epochs, picks, config=DEFAULT_CONFIG)). Use dataclass default factories for arrays (e.g., z_thresholds) to avoid mutable defaults.
Phase 3 – API cleanup & legacy deprecation
1.	Deprecate legacy metrics – Clearly document which legacy metrics are deprecated and provide a migration path. If certain metrics are rarely used, move them behind an optional flag or remove them after a deprecation period.
2.	Remove quarantined code – After verifying no external users rely on outside_annotation and other dead modules, remove them entirely. Publish release notes indicating breaking changes.
3.	Simplify module boundaries – Split large modules (e.g., kinematic_features.py) into cohesive subpackages. Consider blink_features/kinematics/__init__.py that exposes the public API, and separate modules for helpers, metrics and extractors.
4.	Enhance tests – Add parameterised tests for the shared compute skeleton ensuring that style/window logic works across modalities. Provide fixtures for various BlinkerConfig settings to test configurability. Increase coverage for optional feature families and ensure that missing dependencies result in skipped tests rather than errors.
Design proposal: shared compute skeleton
from typing import Callable, Dict, Iterable, List, Sequence, Mapping
import pandas as pd

def compute_features(
    epochs: mne.Epochs,
    picks: Sequence[str] | None,
    metrics_by_style: Dict[str, Sequence[str]],
    compute_func: Callable[[str, np.ndarray, float, str], Mapping[str, float]],
    config: BlinkerConfig,
    *,
    family: str,
) -> pd.DataFrame:
    """Shared orchestration logic for blink feature extraction.

    Parameters
    ----------
    epochs : mne.Epochs
        MNE epochs object with metadata.
    picks : list of str or None
        Channels to process; if None, default channels are resolved.
    metrics_by_style : dict
        Maps style names to the list of metric names for that style.
    compute_func : callable
        Function called as `compute_func(style, segment, sfreq, modality)` to
        compute metric values for a single segment.
    config : BlinkerConfig
        Configuration object providing thresholds and constants.
    family : str
        Name of the feature family (e.g., ``"morphology"``).

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed like ``epochs`` with namespaced feature columns.
    """
    # Resolve sampling frequency and channels
    sfreq = float(epochs.info['sfreq'])
    ch_names, channel_data, index, n_epochs, n_times = prepare_epoch_channel_data(
        epochs=epochs, picks=picks, sfreq=sfreq
    )
    # Infer modalities and styles using shared helpers
    modality_map = {ch: infer_modality(ch, epochs.info) for ch in ch_names}
    styles_by_modality = {
        mod: available_styles(tuple(epochs.metadata.columns), mod)
        for mod in set(modality_map.values())
    }
    # Build output columns
    columns = []
    for mod, channels in group_channels_by_modality(modality_map).items():
        for style in styles_by_modality.get(mod, {"base"}):
            for metric in metrics_by_style[style]:
                for stat in config.stat_names:
                    for ch in channels:
                        columns.append(f"{mod}__{style}__{family}__{metric}_{stat}__{ch}")
    # Compute per‑epoch records
    records: List[Dict[str, float]] = []
    for ei in range(n_epochs):
        metadata_row = epochs.metadata.iloc[ei] if isinstance(
            epochs.metadata, pd.DataFrame
        ) else pd.Series(dtype=float)
        record: Dict[str, float] = {}
        for ch, mod in modality_map.items():
            styles = styles_by_modality.get(mod, {"base"})
            for style in styles:
                windows = extract_windows(metadata_row, mod, style, n_times)
                for start_idx, end_idx in windows:
                    segment = channel_data[ch]['raw'][ei][start_idx:end_idx]
                    stats = compute_func(style, segment, sfreq, mod)
                    for metric, values in stats.items():
                        for stat_name, val in values.items():
                            col_name = f"{mod}__{style}__{family}__{metric}_{stat_name}__{ch}"
                            record[col_name] = val
        records.append(record)
    return pd.DataFrame.from_records(records, index=index, columns=columns)
This skeleton abstracts the common loop structure. Feature families implement metrics_by_style and compute_func. Configuration values (stat names, thresholds, etc.) come from the BlinkerConfig object. Adopting this design will make it straightforward to add new feature families or modify the style/metric mapping without touching the orchestration code.
Config strategy
•	Define a BlinkerConfig dataclass containing:
•	stat_names: tuple of statistic names (default ("mean", "std", "cv")).
•	Thresholds (e.g., base_fraction, shut_amp_fraction, p_avr_threshold).
•	Arrays such as z_thresholds using field(default_factory=lambda: np.array(...)) to avoid mutable defaults.
•	Optionally, sub‑configs per feature family (e.g., wavelet levels).
•	Replace module‑level constants (e.g., _DEFAULT_WAVEFORM_PARAMS) with references to the config. Pass the config into compute functions so that thresholds are not hidden global state. Provide a global DEFAULT_CONFIG instance in default_setting.py so existing callers continue to work.
•	Consider loading configuration from a YAML or JSON file for user customisation, but maintain a programmatic API for testability.
Test strategy
•	Ensure that all tests import pyblinker without optional dependencies installed. Wrap optional imports in try/except and skip the corresponding tests with informative messages when dependencies are missing.
•	Parameterise tests on modalities and styles. For example, run the shared compute skeleton with different BlinkerConfig settings (e.g., varying stat_names or thresholds) to ensure flexibility.
•	Add unit tests for the shared helpers (available_styles, extract_windows, modality inference). These tests should cover corner cases such as missing metadata columns, out‑of‑order start/end keys and non‑standard style names.
•	Write tests that compare the output of the refactored feature extractors against the pre‑refactor baseline to guard against behavioural changes. Use small synthetic signals where possible to avoid large .fif files.
Risks and mitigations
Risk	Mitigation
Breaking the public API (e.g., column naming conventions)	Preserve the naming pattern {modality}__{style}__{family}__{metric}_{stat}__{channel}. Provide alias columns or deprecation warnings where names change (e.g., _add_legacy_ear_channel_aliases).
Removing legacy metrics used by downstream consumers	Deprecate metrics in phases and document replacements. Provide feature flags to include legacy metrics until clients migrate.
Configuration migration causing different default behaviour	Ensure that default values in BlinkerConfig match the current hard‑coded constants (e.g., shut_amp_fraction = 0.9 and DEFAULT_PARAMS['shut_amp_fraction']). Write regression tests for default settings.
Optional dependencies not installed	Guard optional imports and skip feature families requiring them. Provide installation instructions or vendor the dependency.
Performance overhead from abstraction	Profile the shared compute skeleton; use vectorised NumPy operations where possible to minimise Python loops. Consider caching style windows per epoch to avoid recomputation.
