# Changelog

## [Unreleased]

### Added
- Consolidated `pyblinker.utils` into focused modules (`annotation_utils`, `metadata_utils`, `epoch_utils`, `io_utils`, `refinement_utils`, `report_utils`, `statistics_utils`, `velocity_utils`, `dict_utils`, `iter_utils`, `string_utils`).
- Added `docs/utils_inventory.md` capturing the inventory of utility callables and usage counts.
- Added `docs/utils_migration_mapping.md` documenting legacy-to-new module mappings and deprecation plan.

### Deprecated
- Legacy modules in `pyblinker.utils` now re-export from the canonical implementations and emit `DeprecationWarning`. They will be removed after one minor release.

### Migration
- Update imports to the new modules listed above. Internal code already targets the canonical modules; external consumers should migrate before the next minor release.
