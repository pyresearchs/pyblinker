Here’s a clean refactor plan to pull **all “column header / column list creation” logic** out of `pyblinker/blink_features/morphology/epoch_features.py` into a dedicated module under `pyblinker/blink_features/morphology/`.

## What is “column header creation” in the current file?

In `epoch_features.py`, these parts directly **construct output column names** (or derive the pieces needed to do so):

1. **Legacy column naming helper + maps**

* `_LEGACY_METRIC_STYLE_MAP`, `_LEGACY_MORPHOLOGY_METRICS`, `_DURATION_STYLE_MAP` and `rename_metric_column_name(...)`

2. **Pre-building the full output schema (column list)**

* `_build_output_columns(...)` builds every `"{mod}__{style}__morphology__{metric}_{stat}__{ch}"` column and adds legacy variants

3. **Determining “metrics for a style” used in schema building**

* `_metrics_for_style(...)` appends `duration` and applies method suffix rules
  (uses `MORPHOLOGY_METRIC_STEMS` from `core_metrics.py` )

4. **Per-metric stat column naming during record write**

* `_write_metric_stats_to_record(...)` constructs the per-stat column string

5. **Legacy stat columns naming**

* `build_legacy_morphology_stat_features(...)` uses `rename_metric_column_name(...)` per stat

6. **Legacy EAR alias columns**

* `_add_legacy_ear_channel_aliases(...)` derives alias column names (uppercased channel token)

---

## Suggested new filename (in `pyblinker/blink_features/morphology/`)

### ✅ Recommended: `column_headers.py`

Reason: it matches your intent literally (“header creation”), and keeps it clearly scoped to naming/schema rather than computation.

(Alternatives that also work well: `output_schema.py`, `feature_columns.py` — but `column_headers.py` is the most direct.)

---

## Proposed contents of `column_headers.py`

Move/centralize **only naming + schema** concerns:

### A) Constants / maps (moved as-is)

* `DEFAULT_STATS = ("mean", "std", "cv")` (or keep `_STATS` in `epoch_features.py` but pass it in)
* `_LEGACY_MORPHOLOGY_METRICS`
* `_LEGACY_METRIC_STYLE_MAP`
* `_DURATION_STYLE_MAP`

### B) Small pure helpers (single responsibility)

* `make_morphology_stat_column(modality, style, metric, stat, channel) -> str`

    * wraps the f-string currently embedded in `_write_metric_stats_to_record`
* `rename_metric_column_name(modality, metric, stat_name, channel_name) -> str`

    * move as-is (legacy naming)
* `metrics_for_style(style, metric_stems=MORPHOLOGY_METRIC_STEMS, all_methods=ALL_METHODS) -> list[str]`

    * extracted from `_metrics_for_style`
* `metric_method_for_style(style, all_methods=ALL_METHODS) -> str`

    * extracted from `_metric_method_for_style`

### C) Schema builder

* `build_morphology_output_columns(modality_channels, styles_by_modality, *, stats=DEFAULT_STATS) -> list[str]`

    * extracted from `_build_output_columns`

### D) Legacy alias helper

* `add_legacy_ear_channel_aliases(df: pd.DataFrame) -> pd.DataFrame`

    * move `_add_legacy_ear_channel_aliases` as-is

---

## Step-by-step migration plan

1. **Create** `pyblinker/blink_features/morphology/column_headers.py`

    * Copy in the constants + helpers listed above.
    * Keep dependencies minimal:

        * `from .core_metrics import MORPHOLOGY_METRIC_STEMS`
        * `from .._blink_metrics_shared import ALL_METHODS` (same as current)
        * `import pandas as pd` only if you move the EAR alias function.

2. **Edit `epoch_features.py` to use the new module**

    * Replace the local definitions with imports, e.g.:

        * `from .column_headers import build_morphology_output_columns, make_morphology_stat_column, rename_metric_column_name, metrics_for_style, add_legacy_ear_channel_aliases, _DURATION_STYLE_MAP`
    * In `compute()`, replace:

        * `columns = self._build_output_columns(...)`
          with
        * `columns = build_morphology_output_columns(...)`
    * In `_write_metric_stats_to_record`, replace the inline f-string column creation  with `make_morphology_stat_column(...)`.
    * In `build_legacy_morphology_stat_features`, keep the stats computation but call `rename_metric_column_name(...)` from the new module .
    * Replace `_add_legacy_ear_channel_aliases(df)` usage with `add_legacy_ear_channel_aliases(df)` .

3. **Keep backward compatibility (optional but recommended)**

    * If anything else imports `rename_metric_column_name` from `epoch_features.py`, you can re-export it:

        * `from .column_headers import rename_metric_column_name  # re-export`
    * Same for `_DURATION_STYLE_MAP` if it’s used outside.

4. **Add/adjust a lightweight test**

    * Snapshot test: given a small `modality_channels` + `styles_by_modality`, assert the produced column list matches what `_build_output_columns` used to return.
    * Test that EAR alias columns are still added (uppercased channel suffix) .

---

## End state (what gets smaller)

After this, `epoch_features.py` focuses on:

* extracting windows + computing metrics + aggregating stats,
  and delegates *all* “how column names look / what columns exist” to `column_headers.py`, including:
* schema planning (`build_morphology_output_columns`)
* per-stat naming (`make_morphology_stat_column`)
* legacy naming (`rename_metric_column_name`)
* EAR alias naming (`add_legacy_ear_channel_aliases`)

