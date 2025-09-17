# Changelog

## [Unreleased]

### Added
- Consolidated `pyblinker.utils` into focused modules (`annotation_utils`, `metadata_utils`, `epoch_utils`, `io_utils`, `refinement_utils`, `report_utils`, `statistics_utils`, `velocity_utils`, `dict_utils`, `iter_utils`, `string_utils`).
- Added `docs/utils_inventory.md` capturing the inventory of utility callables and usage counts.
- Added `docs/utils_migration_mapping.md` documenting legacy-to-new module mappings and deprecation plan.

### Deprecated
- None.

### Removed
- Removed the deprecated `pyblinker.utils` shim modules now that the migration window has elapsed.

### Migration
- Update imports to the new modules listed above. Internal code already targets the canonical modules; external consumers must use the canonical modules now that the legacy shims have been removed.
