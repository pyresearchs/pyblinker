The legacy morphology feature names were constructed by concatenating five parts with double-underscores (`__`) as separators:

1. **`modality`** – the signal source (e.g., `eeg`)
2. **`style`** – the morphology group (e.g., `zero`, `base`, `tent`, `half`, `peak`, `inter_blink`)
3. **Fixed token** – the literal string `morphology`
4. **`metric` + `stat_name`** – the metric key (e.g., `duration_zero`) followed by an underscore and the statistic suffix (e.g., `mean`, `std`, `cv`), producing `duration_zero_mean`, `duration_zero_std`, `duration_zero_cv`, etc.
5. **`channel_name`** – the channel identifier (e.g., `EEG-E8`)

Concretely, the final column name string is built as:

```python
col = f"{modality}__{style}__morphology__{metric}_{stat_name}__{channel_name}"
```

So for:

* `modality="eeg"`
* `style="zero"`
* `metric="duration_zero"`
* `stat_name in {"mean","std","cv"}`
* `channel_name="EEG-E8"`

you get:

* `eeg__zero__morphology__duration_zero_mean__EEG-E8`
* `eeg__zero__morphology__duration_zero_std__EEG-E8`
* `eeg__zero__morphology__duration_zero_cv__EEG-E8`

In short: **old vs. new naming differences come from changing any of these components** (most commonly the `style` and/or `metric`), but the assembly rule stays the same:
`modality + style + "morphology" + metric_stat + channel`, joined with `__`, with `metric` and `stat_name` joined by `_`.
