# Changelog

## [Unreleased]

### Fixed
- Refactor morphology epoch feature extraction to support EAR-only metadata styles (for example `th_point` and `th_interpolation`) by resolving generic `start__<style>__<modality>`/`end__<style>__<modality>` frame windows and mapping custom styles to the base morphology metric key space.
- Add EAR morphology output compatibility aliases that expose uppercase channel-suffix columns (for example `__EAR-AVG_EAR`) expected by existing EAR-only tests, while preserving existing EEG column names and legacy EEG-only metrics.

### Changed
- Refactor kinematic epoch extraction to discover generic modality styles from `start__<style>__<modality>`/`end__<style>__<modality>` metadata, enabling EAR-only kinematic windows while preserving EEG style handling.
- Add EAR interpolation compatibility aliases for legacy kinematic column names used by downstream checks.
- Further refactor kinematic epoch aggregation by extracting window-level segment metric computation and style-stat writing into dedicated helpers in `pyblinker/blink_features/kinematics/kinematic_features.py`.
- Refactor `pyblinker/blink_features/kinematics/kinematic_features.py` by splitting extended legacy kinematic calculations into dedicated helper functions to reduce nesting and improve readability/debugging.
- Add an inline code comment before legacy kinematic computation in the epoch aggregation loop: `calculate the legacy kinematics features`.
- Switch kinematic epoch window extraction and per-window slicing to use frame-based `start__...`/`end__...` metadata directly, avoiding onset/duration-to-sample conversion in this pipeline.
- Update feature documentation to describe frame-window kinematic slicing and the helperized legacy kinematic flow.

## [0.2.2] - 2026-02-20

### Fixed
- Prevent pytest from collecting the comparison-test path helper as a test by renaming `test_file_path(...)` to `get_test_file_path(...)` in `test/blinker_pyblinker_comparison/utils.py`.
- Update all blinker-vs-pyblinker comparison tests to import and call `get_test_file_path(...)`, preserving both per-file pytest execution and aggregate unittest execution via `test/run_all_tests.py`.

## [0.2.1] - 2026-02-20

### Fixed
- Make `test/blinker_pyblinker_comparison` path handling discovery-safe by resolving test fixture files relative to the test package instead of the process working directory.
- Clean up comparison test imports and shared path utilities so the comparison tests execute consistently both via `python -m unittest <module>` and through `python test/run_all_tests.py`.

## [0.2.0] - 2026-02-20

### Added
- Add a dedicated `test/README.md` documenting supported single-module and batch test commands, with `python -m unittest <dotted.module>` as the recommended single-test workflow.

### Changed
- Rebuild `test/run_all_tests.py` to restore runnable discovery, enforce deterministic ordering by test id, and print clear per-module plus per-test execution progress in CI-friendly output.
- Keep batch discovery rooted at `test/` with repository root as `top_level_dir`, and ensure repository-root import path setup is applied consistently before discovery/runs.

## [0.1.11] - 2026-02-12

### Fixed
- Fix morphology style-window extraction to prioritize frame-based landmark boundaries (`start__...`/`end__...`) over onset/duration metadata so per-window metrics no longer collapse to empty windows when duration fields are zero.
- Update morphology per-window metric extraction to slice `signal[start_frame:end_frame]` directly from frame windows (without round-tripping through onset/duration conversion).
- Add regression coverage ensuring `zero` style windows are built from landmark frame indices when present.

## [0.1.10] - 2026-02-11

### Changed
- Refactor kinematic epoch extraction to discover generic modality styles from `start__<style>__<modality>`/`end__<style>__<modality>` metadata, enabling EAR-only kinematic windows while preserving EEG style handling.
- Add EAR interpolation compatibility aliases for legacy kinematic column names used by downstream checks.
- Move inter-blink max velocity epoch assertions into kinematics coverage and keep morphology epoch extraction focused on morphology-only legacy fields.
- Expand EEG-only kinematic and morphology aggregation tests to explicitly assert newly exposed kinematic sign-split ratios/side velocities and legacy morphology timing/peak/inter-blink fields.
- Refactor epoch kinematic extraction to expose per-blink sign-split amplitude/velocity ratios and average side velocities via shared kinematics core-metric utilities.
- Refactor epoch morphology extraction to expose peak time/max and inter-blink max-velocity legacy outputs using shared morphology/kinematics core-metric utilities for parity with BlinkProperties comparisons.

### Fixed
- Tighten MATLAB parity in `get_blink_statistic` by converting zero-crossing blink boundaries with nearest-integer semantics (`rint`) before building the inclusive blink mask.
- Strengthen comparison coverage by using `atol = 1e-4` in `test_a2_stat.py`, confirming stable statistics under a stricter tolerance.

## [0.1.9] - 2026-02-11

### Fixed
- Correct `get_blink_statistic` to report `number_blinks` from the full blink-fit table (matching MATLAB `extractBlinks`), while still excluding invalid rows for downstream quality/cutoff statistics.
- This restores MATLAB-parity for blink statistics and makes the blinker comparison tests pass consistently when run individually or through the aggregate runner.

## [0.1.8] - 2026-02-11

### Fixed
- Align `get_blink_statistic` with MATLAB `extractBlinks` semantics by dropping invalid blink-fit rows before all downstream statistics and mask calculations.
- Match MATLAB blink amplitude ratio masking behavior with inclusive zero-crossing ranges and robust handling of empty inside/outside positive-signal samples.
- Return deterministic NaN-based statistics when fewer than two top-threshold blink fits are available, mirroring MATLAB early-exit behavior for cutoff statistics.

## [0.1.7] - 2026-02-11

### Fixed
- Stabilize FitBlinks correlation outputs (`leftR2`, `rightR2`) for MATLAB comparison by normalizing returned R values to single-precision representation.
- Relax strict floating-point equality in `step_b_fitblink.py` to MATLAB-appropriate absolute tolerance (`atol=1e-6`) to avoid false failures from numeric noise.


## [0.1.6] - 2026-02-11

### Fixed
- Restore MATLAB-consistent left half-height landmark indexing in FitBlinks geometry (base and zero half-height frames).
- Restore MATLAB-consistent inter-blink max-velocity fields used by BlinkProperties comparison outputs.
- Guard amplitude-velocity ratio velocity-index bounds to avoid out-of-range indexing on short boundary blinks.


## [0.1.5] - 2026-02-11

### Changed
- Refactor kinematic epoch extraction to discover generic modality styles from `start__<style>__<modality>`/`end__<style>__<modality>` metadata, enabling EAR-only kinematic windows while preserving EEG style handling.
- Add EAR interpolation compatibility aliases for legacy kinematic column names used by downstream checks.
- Refactor blink-position detection and blink-property extraction flows into MATLAB-aligned helper stages for clearer parity-oriented maintenance.

### Fixed
- Align blink start/end candidate scanning and minimum-separation masking with MATLAB `getBlinkPositions` semantics.
- Correct baseline and half-height landmark indexing offsets used during FitBlinks geometry calculations to match MATLAB frame conventions.
- Fix inter-blink maximum-velocity interval computation so BlinkProperties uses MATLAB-style successive-peak differences.


## [0.1.4] - 2026-01-20

### Changed
- Refactor kinematic epoch extraction to discover generic modality styles from `start__<style>__<modality>`/`end__<style>__<modality>` metadata, enabling EAR-only kinematic windows while preserving EEG style handling.
- Add EAR interpolation compatibility aliases for legacy kinematic column names used by downstream checks.
- Expand blink-property metric definitions and MATLAB-to-Python column mapping documentation.

### Fixed
- Align blink-property timing, shut-time defaults, and inter-blink velocity intervals with MATLAB to support row-by-row comparisons.

## [0.1.3] - 2026-01-19

### Changed
- Refactor kinematic epoch extraction to discover generic modality styles from `start__<style>__<modality>`/`end__<style>__<modality>` metadata, enabling EAR-only kinematic windows while preserving EEG style handling.
- Add EAR interpolation compatibility aliases for legacy kinematic column names used by downstream checks.
- Render FitBlinks tutorial waveforms as scatter points with a faint line trace behind them and pad report windows on the left by 10 samples for clearer landmark visibility.

## [0.1.1] - 2026-01-19

### Fixed
- Resolve FitBlinks tutorial input paths relative to the repository and write reports to the tutorial_outputs folder.

## [0.1.0] - 2026-01-18

### Added
- Add MATLAB FitBlinks HTML reporting utilities to compare MATLAB output with Python-derived landmarks using consistent plotting and captions.

## [0.0.74] - 2026-01-17

### Fixed
- Keep FitBlinks y-intersect comparison values in signal units without indexing offsets.

## [0.0.73] - 2026-01-17

### Fixed
- Adjust FitBlinks comparison indexing so y-intersect values remain in signal units.

## [0.0.72] - 2026-01-17

### Fixed
- Align FitBlinks comparison indexing for intersection outputs without extra y-intersect offset.

## [0.0.71] - 2026-01-17

### Fixed
- Emit NaN half-height landmarks when thresholds are never crossed to match MATLAB fit behavior.

## [0.0.70] - 2026-01-17

### Fixed
- Correct right-base index handling in FitBlinks to match MATLAB downstroke minima behavior.

## [0.0.69] - 2026-01-17

### Fixed
- Align FitBlinks indexing, baseline selection, and line-intersection handling with MATLAB to preserve blink counts and fit outputs.

## [0.0.68] - 2026-01-16

### Fixed
- Match MATLAB blink position separation handling by allowing candidates exactly at the minimum separation threshold.

### Changed
- Refactor kinematic epoch extraction to discover generic modality styles from `start__<style>__<modality>`/`end__<style>__<modality>` metadata, enabling EAR-only kinematic windows while preserving EEG style handling.
- Add EAR interpolation compatibility aliases for legacy kinematic column names used by downstream checks.
- Refactor blink position detection into vectorized helpers for improved efficiency without altering behavior.

## [0.0.67] - 2026-01-15

### Changed
- Refactor kinematic epoch extraction to discover generic modality styles from `start__<style>__<modality>`/`end__<style>__<modality>` metadata, enabling EAR-only kinematic windows while preserving EEG style handling.
- Add EAR interpolation compatibility aliases for legacy kinematic column names used by downstream checks.
- Compute morphology epoch metrics with shared core duration/shut-time helpers while emitting both legacy flat column names and fully-qualified style-aware names.
- Replace lazy morphology package exports with explicit imports aligned to the kinematics package.
- Add morphology aggregation logging and clarify core metric responsibilities in documentation strings.
- Refactor morphology epoch extraction into helper steps with pipeline-oriented docstrings.

### Added
- EEG-only morphology test coverage for style-aware duration columns and legacy aliases.

## [0.0.66] - 2026-01-14

### Fixed
- Draw EEG epoch landmark markers behind EEG scatter points in the EEG landmark tutorial to keep EEG-E8 dots on top without changing marker sizes.
