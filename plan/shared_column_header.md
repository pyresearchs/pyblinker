Here’s the **shared refactor plan** I’d hand to an implementation agent after reading the 3 plans + the current Python naming/schema code.

## What’s common across Energy / Kinematics / Morphology

All three feature families are independently doing the same *schema* work:

1. **Same column “shape” (token layout)**
   They all emit columns shaped like:

`{modality}__{style}__{feature_group}__{metric}_{stat}__{channel}`

* Energy builds these in `_make_columns(...)` and also repeats the same f-string during record writing.
* Kinematics builds them via a `column_set` loop.
* Morphology builds them in `_build_output_columns(...)` and repeats the f-string in `_write_metric_stats_to_record(...)`.

2. **Same statistics axis**
   All use the same `("mean", "std", "cv")` stat set (Energy `_STATS`, Kinematics `_STATS`, Morphology `_STATS`).

3. **Same refactor target described in the plans**
   The kinematics plan explicitly defines the reusable surface: `FEATURE_GROUP`, `STATS`, `metrics_for_style`, `make_stat_column`, `build_output_columns`, `add_legacy_alias_columns`.
   It also suggests an optional tiny shared helper to reduce duplication.
   Morphology’s plan identifies the same extraction set (schema builder + stat-column formatter + legacy naming/alias pieces).

4. **Channel-label quirks (domain-specific, but pattern repeats)**

* Energy uses a modality rule for channel label casing: `eog` keeps original, others uppercase.
* Morphology adds *alias columns* by uppercasing the channel suffix for `ear__...` columns.
* Kinematics has an EAR interpolation alias helper (present but currently commented at callsite).

So: **the “column header / column list creation” logic is structurally the same**, even though each domain has extra rules.

---

## Shared design decision

### ✅ Keep **domain modules** (`energy/column_headers.py`, `kinematics/column_headers.py`, `morphology/column_headers.py`)

They hold domain-specific knobs (metric lists, legacy maps, alias policies).

### ✅ Add **one cross-domain helper**: `pyblinker/blink_features/utils/column_headers_common.py`

This eliminates the repeated f-strings and repeated schema loops across 3 domains.

This aligns with the “optional shared base helper” idea already in the kinematics plan.

---

## New shared helper module

### File: `pyblinker/blink_features/utils/column_headers_common.py`

**Responsibilities (generic only):**

1. **Stat-column formatter**

```python
def make_stat_column(*, modality: str, style: str, feature_group: str,
                     metric: str, stat: str, channel: str) -> str:
    return f"{modality}__{style}__{feature_group}__{metric}_{stat}__{channel}"
```

2. **Schema builder loop**

```python
def build_output_columns(*, modality_channels: dict[str, list[str]],
                         styles_by_modality: dict[str, set[str]],
                         feature_group: str,
                         metrics_for_style: Callable[[str], Sequence[str]],
                         stats: Sequence[str],
                         channel_label: Callable[[str, str], str] = lambda ch, mod: ch,
                         extra_columns: Callable[[str, list[str]], Iterable[str]] | None = None,
                         default_styles: set[str] = {"base"}) -> list[str]:
    # loops mod->styles->metrics->stats->channels and returns sorted unique columns
```

* `channel_label(ch, modality)` handles Energy’s uppercasing rule cleanly.
* `extra_columns(modality, channels)` supports Morphology legacy columns (EEG/EOG primary channel) without polluting the generic loop.

---

## Per-domain modules and how they use the shared helper

### A) Energy

#### New file: `pyblinker/blink_features/energy/column_headers.py`

Exports (same public surface as the other domains):

* `FEATURE_GROUP = "energy"`
* `STATS = ("mean", "std", "cv")`
* `METRICS = (...)` (move `_METRICS` here)
* `metrics_for_style(style) -> list[str]` → returns `list(METRICS)` (energy metrics don’t change by style today)
* `channel_label(channel, modality)` → replicate `_feature_channel_name` rule (eog keeps case, else upper).
* `make_stat_column(...)` → wrapper over common `make_stat_column(feature_group="energy")`
* `build_output_columns(modality_channels, styles_by_modality, ...)` → wrapper over common builder
* `add_legacy_alias_columns(df)` → **no-op** (just `return df`) for symmetry with other domains

#### Update `energy_features.py`

* Replace `_STATS`, `_METRICS`, `_feature_channel_name`, `_make_columns` (schema-only pieces).
* Replace the record-write f-string with `make_stat_column(...)`.
* Keep `_normalize_styles_for_modality(...)` in energy_features.py (it’s not “header creation”; it’s semantic style selection).

---

### B) Kinematics

#### New file: `pyblinker/blink_features/kinematics/column_headers.py`

Follow the kinematics plan API exactly:

* `FEATURE_GROUP = "kinematic"`
* `STATS = ("mean", "std", "cv")`
* `EXTENDED_METRICS = (...)` (from `_EXTENDED_KINEMATIC_METRICS`)
* `metrics_for_style(style)` (from `_metrics_for_style`)
* `make_stat_column(...)` (wrapper around common helper)
* `build_output_columns(...)` (wrapper around common builder; include `EXTENDED_METRICS`)
* `add_legacy_alias_columns(df)` (move `_add_legacy_ear_interpolation_aliases`)

This is literally what the plan calls out.

#### Update `kinematic_features.py`

* Replace the `column_set` loop with `build_output_columns(...)`.
* Replace per-stat f-string creation inside `_write_style_stats_into_record` via `make_stat_column(...)` (same idea as plan).
* If aliasing is still optional, keep the call commented but route to `add_legacy_alias_columns`.

---

### C) Morphology

#### New file: `pyblinker/blink_features/morphology/column_headers.py`

Exports:

* `FEATURE_GROUP = "morphology"`
* `STATS = ("mean", "std", "cv")`
* `metrics_for_style(style)` (from `_metrics_for_style`)
* `metric_method_for_style(style)` (if you want parity with current code)
* `make_stat_column(...)` (wrapper around common helper; replaces the f-string used in `_write_metric_stats_to_record`)
* `build_output_columns(...)` (wrap common builder + inject morphology legacy columns logic via `extra_columns(...)`)
* Keep Morphology-specific extra exports:

    * `LEGACY_METRIC_STYLE_MAP`, `LEGACY_MORPHOLOGY_METRICS`, `DURATION_STYLE_MAP`
    * `rename_metric_column_name(...)` (legacy naming helper)
* `add_legacy_alias_columns(df)` moves `_add_legacy_ear_channel_aliases`

This matches the morphology plan’s inventory of what should move.

#### Update `epoch_features.py` (morphology)

* Replace `self._build_output_columns(...)` with `column_headers.build_output_columns(...)`.
* Replace the f-string in `_write_metric_stats_to_record` with `make_stat_column(...)`.
* Keep `build_legacy_morphology_stat_features` logic but import `rename_metric_column_name` from the new module.
* Apply `add_legacy_alias_columns(df)` after `frame_from_records`.

---

## Implementation order (recommended)

1. Add `utils/column_headers_common.py` (formatter + schema builder).
2. Add `energy/column_headers.py`, refactor `energy_features.py` to use it.
3. Add `kinematics/column_headers.py`, refactor `kinematic_features.py`.
4. Add `morphology/column_headers.py`, refactor `epoch_features.py`.
5. Run tests; then add snapshot tests (below).

---

## Tests / quality gates (to prevent silent schema drift)

1. **Snapshot schema tests** (one per domain):

* Build a small fake `modality_channels` and `styles_by_modality` and assert `build_output_columns(...)` exactly matches the old output ordering.

    * Kinematics baseline should match the `column_set` loop output.
    * Morphology baseline should match `_build_output_columns`, including legacy metrics behavior.
    * Energy baseline should match `_make_columns` ordering and channel label casing rule.

2. **Alias behavior tests**

* Morphology: assert uppercase EAR alias columns are created.
* Kinematics: assert interpolation alias mapping unchanged (if enabled).

3. **Record-key consistency**

* Ensure record-writing uses `make_stat_column` everywhere so you don’t get “columns exist but never written” (or vice versa). Energy currently repeats the f-string both for schema and record writing.

---

