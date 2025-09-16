# pyblinker.utils module migration

The table below maps legacy utility modules to their new canonical homes. Deprecated shims remain for one minor release and emit `DeprecationWarning` when imported.

| Legacy module | Replacement(s) |
| --- | --- |
| `pyblinker.utils.misc` | `pyblinker.utils.annotation_utils` |
| `pyblinker.utils.blink_metadata` | `pyblinker.utils.metadata_utils` |
| `pyblinker.utils.blink_windows` | `pyblinker.utils.metadata_utils` |
| `pyblinker.utils.blink_refinement_helpers` | `pyblinker.utils.dict_utils` |
| `pyblinker.utils.segments` | `pyblinker.utils.epoch_utils` |
| `pyblinker.utils.epochs` | `pyblinker.utils.epoch_utils`, `pyblinker.utils.io_utils`, `pyblinker.utils.report_utils` |
| `pyblinker.utils.refine_util` | `pyblinker.utils.refinement_utils` |
| `pyblinker.utils.refinement` | `pyblinker.utils.refinement_utils` |
| `pyblinker.utils.raw_preprocessing` | `pyblinker.utils.io_utils` |
| `pyblinker.utils.report` | `pyblinker.utils.report_utils` |
| `pyblinker.utils.blink_statistics` | `pyblinker.utils.statistics_utils` |
| `pyblinker.utils.velocity` | `pyblinker.utils.velocity_utils` |

New supporting modules introduced during the consolidation:

- `pyblinker.utils.annotation_utils`
- `pyblinker.utils.string_utils`
- `pyblinker.utils.iter_utils`
- `pyblinker.utils.dict_utils`
- `pyblinker.utils.metadata_utils`
- `pyblinker.utils.epoch_utils`
- `pyblinker.utils.io_utils`
- `pyblinker.utils.refinement_utils`
- `pyblinker.utils.report_utils`
- `pyblinker.utils.statistics_utils`
- `pyblinker.utils.velocity_utils`

All internal imports now target the new canonical modules.
