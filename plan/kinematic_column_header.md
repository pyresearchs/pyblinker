Below is a refactor plan for **KinematicBlinkFeatureExtractor** that mirrors what you’re doing for morphology, and a **shared “column_headers.py structure/design”** so both kinematics + morphology follow the same pattern across the repo.

---

## 1) What counts as “column header / column list creation” in `kinematic_features.py`

In `pyblinker/blink_features/kinematics/kinematic_features.py`, the column-related responsibilities are currently spread across:

1. **Global schema knobs**

* `_STATS = ("mean", "std", "cv")`
* `_EXTENDED_KINEMATIC_METRICS = (...)` (these are extra metric names that become columns)

2. **Metric-name generation for a style**

* `_metrics_for_style(style)` constructs metric names with a style suffix except for `KINEMATIC_METRICS_NO_STYLE`.
  (uses `KINEMATIC_METRIC_STEMS` and `KINEMATIC_METRICS_NO_STYLE` from `core_metrics.py`. )

3. **Column list creation for the output DataFrame**

* In `compute()`: the big `column_set` loop that builds `f"{mod}__{style}__kinematic__{metric}_{stat}__{ch}"` for every modality/style/metric/stat/channel.

4. **Per-stat column name formatting during record write**

* `_write_style_stats_into_record(...)` builds `column = f"{modality}__{style}__kinematic__{metric_name}_{stat_name}__{channel_name}"`.

5. **Legacy alias columns**

* `_add_legacy_ear_interpolation_aliases(df)` creates additional alias column names for backwards compatibility.

Those are the parts you should move.

---

## 2) New module: `pyblinker/blink_features/kinematics/column_headers.py`

Create:

* `pyblinker/blink_features/kinematics/column_headers.py`

This mirrors the morphology approach and keeps kinematic_features.py focused on *computation*, not naming/schema.

### What to move into `kinematics/column_headers.py`

**A) Constants**

* `FEATURE_GROUP = "kinematic"` (new, for consistency)
* `STATS = ("mean", "std", "cv")` (move from `_STATS`)
* `EXTENDED_METRICS = (...)` (move from `_EXTENDED_KINEMATIC_METRICS`)

**B) Metric naming**

* `metrics_for_style(style: str) -> list[str]`

    * move `_metrics_for_style` logic unchanged.

**C) Column name formatter**

* `make_stat_column(modality: str, style: str, metric: str, stat: str, channel: str) -> str`

    * encapsulates the kinematic f-string currently in `_write_style_stats_into_record`.
    * This should match the morphology formatter shape exactly (only `FEATURE_GROUP` differs).

**D) Output schema builder**

* `build_output_columns(modality_channels, styles_by_modality, *, stats=STATS, include_extended=True) -> list[str]`

    * lifts the `column_set` construction loop from `compute()` and returns `sorted(columns)`.

**E) Legacy aliases**

* `add_legacy_alias_columns(df: pd.DataFrame) -> pd.DataFrame`

    * move `_add_legacy_ear_interpolation_aliases` as-is, but rename to the consistent API name.

---

## 3) How to update `kinematic_features.py` after extracting

1. **Import the new naming API**

* `from .column_headers import (STATS, EXTENDED_METRICS, metrics_for_style, build_output_columns, make_stat_column, add_legacy_alias_columns)`

2. **Replace the schema creation block**

* Replace the inlined `column_set` logic in `compute()` with:

    * `columns = build_output_columns(modality_channels, styles_by_modality, stats=STATS, include_extended=True)`

3. **Replace per-stat f-string in `_write_style_stats_into_record`**

* Replace:

    * `column = f"{modality}__{style}__kinematic__{metric_name}_{stat_name}__{channel_name}"`
* With:

    * `column = make_stat_column(modality, style, metric_name, stat_name, channel_name)`

4. **Replace `_metrics_for_style(style)` usage**

* Swap calls to the imported `metrics_for_style(style)`

5. **Move legacy alias function**

* Delete `_add_legacy_ear_interpolation_aliases` from this file and call:

    * `df = add_legacy_alias_columns(df)`
    * (only if you actually need it enabled; right now the call is commented in your file. )

---

## 4) Repository-wide consistency: a shared structure/design for BOTH `column_headers.py`

To make `morphology/column_headers.py` and `kinematics/column_headers.py` consistent, enforce the *same public surface* and file layout.

### ✅ Required public API (same names in both modules)

Each `column_headers.py` must export:

* `FEATURE_GROUP: str`
* `STATS: tuple[str, ...]`
* `metrics_for_style(style: str) -> list[str]`
* `make_stat_column(modality: str, style: str, metric: str, stat: str, channel: str) -> str`
* `build_output_columns(modality_channels, styles_by_modality, *, stats=STATS, **kwargs) -> list[str]`
* `add_legacy_alias_columns(df: pd.DataFrame) -> pd.DataFrame`

### ✅ Required internal section order (same structure)

1. **Module docstring**
2. **Imports**
3. **Public constants** (`FEATURE_GROUP`, `STATS`, domain-specific extras like `EXTENDED_METRICS`)
4. **Metric naming** (`metrics_for_style`)
5. **Column formatting** (`make_stat_column`)
6. **Schema builder** (`build_output_columns`)
7. **Legacy aliases** (`add_legacy_alias_columns`)
8. **`__all__`**

### Domain-specific differences (allowed)

* Morphology’s `metrics_for_style` depends on morphology stems/method rules; kinematics’ depends on `KINEMATIC_METRIC_STEMS` and `KINEMATIC_METRICS_NO_STYLE`.
* Morphology may include “legacy morphology metric/style maps”; kinematics includes `EXTENDED_METRICS` and the EAR interpolation aliasing.

---

## 5) Optional (but clean) extra consistency: a tiny shared base helper

If you want to reduce duplication further, you can add a small shared helper in something like:

* `pyblinker/blink_features/utils/column_headers_common.py`

Containing only the generic f-string builder:

* `make_stat_column(modality, style, feature_group, metric, stat, channel)`

Then each domain module wraps it with its own `FEATURE_GROUP`. This keeps the *structure identical* while avoiding two copies of the same string formatting logic.


