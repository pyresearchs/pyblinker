# Changelog

## [0.0.67] - 2026-01-15

### Changed
- Compute morphology epoch metrics with shared core duration/shut-time helpers while emitting both legacy flat column names and fully-qualified style-aware names.
- Replace lazy morphology package exports with explicit imports aligned to the kinematics package.

### Added
- EEG-only morphology test coverage for style-aware duration columns and legacy aliases.

## [0.0.66] - 2026-01-14

### Fixed
- Draw EEG epoch landmark markers behind EEG scatter points in the EEG landmark tutorial to keep EEG-E8 dots on top without changing marker sizes.
