# pyblinker.utils inventory


| Module | Shim | Callable | Type | Scope | Signature | Docstring | Reference count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pyblinker/utils/annotation_utils.py | no | create_annotation | function | public | (sblink, sfreq, label) | Convert blink spans into an :class:`mne.Annotations` object. | 17 |
| pyblinker/utils/channel_utils.py | no | _is_ear_channel | function | private | (name) |  | 3 |
| pyblinker/utils/channel_utils.py | no | normalize_picks | function | public | (picks) | Normalize channel picks to a list. | 12 |
| pyblinker/utils/channel_utils.py | no | require_channels | function | public | (data, picks) | Validate that all requested channels exist in the provided data. | 11 |
| pyblinker/utils/channel_utils.py | no | pick_ear_channels_from_info | function | public | (info) | Return indices of EAR-style channels from an :class:`mne.Info`. | 6 |
| pyblinker/utils/channel_utils.py | no | pick_ear_channels_from_raw | function | public | (raw) | Return indices of EAR-style channels from a :class:`mne.io.BaseRaw`. | 6 |
| pyblinker/utils/dict_utils.py | no | append_to_slot | function | public | (slot, value) | Append ``value`` to ``slot`` preserving backwards compatible semantics. | 20 |
| pyblinker/utils/dict_utils.py | no | contains_key | function | public | (container, key) | Return ``True`` if ``key`` exists in ``container``. | 5 |
| pyblinker/utils/dict_utils.py | no | group_by_key | function | public | (items, key) | Group mapping items by ``key`` value. | 5 |
| pyblinker/utils/dict_utils.py | no | update_dict_list | function | public | (target, key, value) | Ensure ``target[key]`` is a list and extend it with ``value``. | 2 |
| pyblinker/utils/epoch_utils.py | no | slice_raw_to_segments | function | public | (raw, epoch_len=30.0) | Slice a continuous :class:`mne.io.BaseRaw` into fixed-length segments. | 6 |
| pyblinker/utils/epoch_utils.py | no | slice_raw_into_mne_epochs | function | public | (raw) | Convert a continuous recording into equally spaced MNE epochs. | 14 |
| pyblinker/utils/epoch_utils.py | no | slice_raw_into_epochs | function | public | (raw) | Slice a raw recording into epochs and count blink annotations. | 17 |
| pyblinker/utils/epoch_utils.py | no | slice_into_mini_raws | function | public | (raw, out_dir) | Slice a raw recording into epochs with optional saving and reporting. | 8 |
| pyblinker/utils/io_utils.py | no | save_epoch_raws | function | public | (segments, times, out_dir) | Save cropped raw segments to disk. | 8 |
| pyblinker/utils/io_utils.py | no | _update_segment_annotations | function | private | (segments, refined) |  | 2 |
| pyblinker/utils/io_utils.py | no | prepare_refined_segments | function | public | (raw, channel) | Load and prepare raw segments with refined blink annotations. | 6 |
| pyblinker/utils/iter_utils.py | no | ensure_list | function | public | (value) |  | 9 |
| pyblinker/utils/iter_utils.py | no | ensure_list | function | public | (value) |  | 9 |
| pyblinker/utils/iter_utils.py | no | ensure_list | function | public | (value) |  | 9 |
| pyblinker/utils/iter_utils.py | no | ensure_list | function | public | (value) |  | 9 |
| pyblinker/utils/iter_utils.py | no | ensure_list | function | public | (value) | Coerce ``value`` to a list. | 9 |
| pyblinker/utils/iter_utils.py | no | ensure_float_list | function | public | (value) | Return a list of floats, gracefully handling ``None`` and ``NaN``. | 6 |
| pyblinker/utils/iter_utils.py | no | iter_chunks | function | public | (iterable, size) | Yield fixed-size chunks from ``iterable``. | 2 |
| pyblinker/utils/metadata_utils.py | no | onset_entry_to_blinks | function | public | (onset) | Convert a ``blink_onset`` metadata entry into blink dictionaries. | 8 |
| pyblinker/utils/metadata_utils.py | no | attach_blink_metadata | function | public | (epochs, blink_df) | Aggregate per-blink properties and merge them into epoch metadata. | 12 |
| pyblinker/utils/metadata_utils.py | no | sample_windows_from_metadata | function | public | (metadata, channel, sfreq, n_times, epoch_index) | Convert blink onset/duration metadata to sample windows. | 9 |
| pyblinker/utils/metadata_utils.py | no | extract_blink_windows | function | public | (metadata_row, channel, epoch_index) | Extract blink onset/duration pairs for a single epoch. | 32 |
| pyblinker/utils/modality.py | no | infer_modality | function | public | (channel_name) | Infer a modality label from a channel name. | 17 |
| pyblinker/utils/open_eye_baseline.py | no | _blinks_from_metadata | function | private | (meta, sfreq) | Convert blink onset/duration metadata to frame spans. | 2 |
| pyblinker/utils/open_eye_baseline.py | no | _compute_features | function | private | (signal, blinks, sfreq) | Compute baseline features for a single-channel epoch. | 5 |
| pyblinker/utils/open_eye_baseline.py | no | compute_open_eye_baseline_features | function | public | (epochs, picks, indices) | Compute averaged baseline features across selected epochs. | 3 |
| pyblinker/segmentation/refinement.py | no | _init_metadata | function | private | (n_epochs, have_eeg, have_eog, have_ear) | Create metadata dict with required (manual) and conditional fields. | 2 |
| pyblinker/segmentation/refinement.py | no | slice_raw_into_mne_epochs_refine_annot | function | public | (raw) | Convert a continuous recording into equally spaced epochs with refinement. | 47 |
| pyblinker/segmentation/refinement.py | no | refine_local_maximum_stub | function | public | (signal_segment, start_rel, end_rel, peak_rel_cvat=None) | Return a crude refinement for local maxima in a signal segment. | 9 |
| pyblinker/segmentation/refinement.py | no | refine_blinks_from_epochs | function | public | (segments, channel) | Refine blink annotations within pre-sliced raw segments. | 8 |
| pyblinker/segmentation/ear.py | no | _empty_interpolated_thresholds | function | private | () | Return default interpolated threshold metadata with NaN/False values. | 2 |
| pyblinker/segmentation/ear.py | no | _refine_ear_blinks_for_epoch | function | private | (segment, blink_starts, blink_ends, sfreq, segmentation_config) | Refine EAR blinks for a single epoch based on segmentation settings. | 3 |
| pyblinker/segmentation/ear.py | no | _append_ear_refinements | function | private | (row_data, refinements, sfreq, n_samp_epoch) | Attach EAR refinement metadata to the epoch metadata frame. | 4 |
| pyblinker/segmentation/ear.py | no | _append_outer_bounds_from_peaks | function | private | (row_data, peaks, key_prefix, n_samp_epoch) | Append blink outer bounds derived from peaks for the given modality. | 4 |
| pyblinker/utils/report_utils.py | no | generate_epoch_report | function | public | (segments, times) | Create a simple report visualizing each segment. | 10 |
| pyblinker/utils/report_utils.py | no | add_blink_plots_to_report | function | public | (epochs) | Add per-blink plots to an :class:`mne.Report` for manual validation. | 9 |
| pyblinker/utils/statistics_utils.py | no | calculate_within_range | function | public | (all_values, best_median, best_robust_std) | Return the count of values within two robust standard deviations. | 7 |
| pyblinker/utils/statistics_utils.py | no | calculate_good_ratio | function | public | (all_values, best_median, best_robust_std, all_x) | Return the fraction of ``all_values`` within the robust range. | 7 |
| pyblinker/utils/statistics_utils.py | no | get_blink_statistic | function | public | (df, z_thresholds, signal=None) | Compute blink statistics for a DataFrame of blink fits. | 13 |
| pyblinker/utils/statistics_utils.py | no | get_good_blink_mask | function | public | (blink_fits, specified_median, specified_std, z_thresholds) | Return mask of good blinks and subset DataFrame based on thresholds. | 13 |
| pyblinker/utils/string_utils.py | no | safe_literal_eval | function | public | (value) | Safely evaluate ``value`` using :func:`ast.literal_eval`. | 5 |
| pyblinker/utils/velocity_utils.py | no | SupportsCoef | class | public | (...) | Protocol describing polynomial-like objects exposing ``coef``. | 3 |
| pyblinker/utils/velocity_utils.py | no | _extract_linear_slope | function | private | (coefficients) | Return the slope component from ``coefficients``. | 2 |
| pyblinker/utils/velocity_utils.py | no | average_velocity | function | public | (coefficients) | Compute the average velocity associated with a linear fit. | 9 |
