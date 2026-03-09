### What’s different between MATLAB Blinker vs Python (your pyblinker implementation)

#### 1) **Where “candidate evaluation” happens**

* **MATLAB:** Treats *candidate signals* (often ICs / candidate components) inside `extractBlinks.m`. It loops over all candidates (`for k = 1:length(params.signalNumbers)`), computes per-candidate quality metrics, then filters down to one.

    * **Main place:** `extractBlinks.m`
* **Python:** Treats *candidate channels* (or candidate signals you iterate over) outside, and you compute everything per channel, store results, then select best channel at the end.

    * **Main place:** your per-channel pipeline + `channel_selection(...)`

**MATLAB equivalent mapping**

* Python “for each channel” loop ≈ MATLAB `for k=1:length(params.signalNumbers)` loop in `extractBlinks.m`.

---

#### 2) **FitBlinks is computed twice in MATLAB, once in Python**

This is the biggest architectural difference.

* **MATLAB (two passes):**

    1. **During candidate selection** in `extractBlinks.m`:

        * calls `fitBlinks(signalData(k).signal, signalData(k).blinkPositions)`
        * uses those fits to compute:

            * `blinkAmpRatio`
            * `bestMedian`, `bestRobustStd`, `cutoff`
            * `goodRatio`
            * `numberGoodBlinks`
        * then chooses **one** candidate signal
    2. **After a signal is chosen**, during finalization in `extractBlinkProperties.m`:

        * calls `fitBlinks(signalData.signal, signalData.blinkPositions)` **again**
        * then reduces blinks (`getGoodBlinkMask`)
        * then computes per-blink properties (`duration*`, `pos/negAmpVelRatio*`, `timeShut*`, etc.)
        * then applies pAVR restriction

So MATLAB’s structure is:
**(fit → selection metrics)** + **(fit again → per-blink properties)**

* **Python (one pass per channel):**

    * You do:

        1. `get_blink_position(...)`
        2. `FitBlinks(...).dprocess()`  ✅ only once
        3. `get_blink_statistic(...)` (selection metrics) ✅ computed from the same fitted df
        4. `get_good_blink_mask(...)` (reduce to good blinks) ✅ same df
        5. `BlinkProperties(...).df` (per-blink properties) ✅ same df
        6. pAVR restriction ✅ same df
    * Then you store results and later select best channel via `channel_selection(...)`.

So Python’s structure is:
**(fit once → selection metrics + per-blink properties in one go)**

**MATLAB equivalent mapping**

* `FitBlinks(...).dprocess()` ≈ MATLAB `fitBlinks(...)`
* Python “do it once” combines logic that MATLAB splits between:

    * `extractBlinks.m` (candidate scoring) **and**
    * `extractBlinkProperties.m` (final blink fits + properties)

---

#### 3) **When blink properties are computed**

* **MATLAB:** Only computes **blink properties** after it already picked the final signal, inside `extractBlinkProperties.m`.
* **Python:** Computes **blink properties for every candidate channel** (because it’s part of your per-channel pipeline), and selection happens afterwards.

**MATLAB equivalent mapping**

* Python `BlinkProperties(...)` ≈ MATLAB per-blink loop inside `extractBlinkProperties.m`.

---

#### 4) **Where the selection filters live**

* **MATLAB:** Selection filters are inline inside `extractBlinks.m` after the loop:

    * amplitude ratio range filter (`blinkAmpRange`)
    * min good blinks (`minGoodBlinks`)
    * good ratio threshold (`goodRatioThreshold`) + keepSignals logic
    * choose max `numberGoodBlinks`
* **Python:** You moved these into a dedicated function `channel_selection(...)` which calls:

    * `filter_blink_amplitude_ratios`
    * `filter_good_blinks`
    * `filter_good_ratio`
    * `select_max_good_blinks`

**MATLAB equivalent mapping**

* Python `channel_selection(...)` ≈ MATLAB “post-loop” block in `extractBlinks.m`:

    * `%% Reduce... blinkAmpRatios...`
    * `%% Find the ones... minGoodBlinks`
    * `%% Now see... goodRatioThreshold`
    * `%% Now pick... maximum number of good blinks`

---

#### 5) **Data products: what gets saved and reused**

* **MATLAB:**

    * Candidate selection keeps summary metrics in `signalData(k)` (amp ratio, goodRatio, etc.)
    * But final blinkFits + blinkProps are recomputed for selected signal in `extractBlinkProperties.m`.
* **Python:**

    * For each channel, you retain:

        * `blink_stats` (summary metrics per channel)
        * `df_out` (final per-blink properties after good mask + pAVR restriction)
    * After selection, you simply pick the stored `df_out` for the winning channel.
      ✅ no second “finalization” recomputation needed.

---

## Updated Python flowchart (with MATLAB equivalent function names embedded)

```plaintext
        +--------------------------------------------------------------+
        |                 Python Blinker (pyblinker) Pipeline           |
        |    (Compute fits + stats + properties ONCE per channel)       |
        |        MATLAB contrast: fitBlinks happens twice               |
        +--------------------------------------------------------------+
                               |
                               v
        +--------------------------------------------------------------+
        | Step 0: Iterate candidate channels/signals                    |
        |   for channel in candidate_channels:                          |
        |                                                              |
        |   MATLAB equivalent: extractBlinks.m                           |
        |     for k = 1:length(params.signalNumbers)                    |
        +--------------------------------------------------------------+
                               |
                               v
        +--------------------------------------------------------------+
        | Step 1: get_blink_position(...)                               |
        |   Output: df (blink positions)                                |
        |                                                              |
        |   MATLAB equivalent: "blinkPositions" feeding fitBlinks       |
        |   (positions used inside extractBlinks.m + extractBlinkProperties.m) |
        +--------------------------------------------------------------+
                               |
                               v
        +--------------------------------------------------------------+
        | Step 2: FitBlinks(...).dprocess()                             |
        |   Output: df = fitblinks.frame_blinks (landmarks per blink)   |
        |                                                              |
        |   MATLAB equivalent: fitBlinks(...)                            |
        |   - called in extractBlinks.m (candidate scoring)             |
        |   - called again in extractBlinkProperties.m (finalization)   |
        |   Python: called ONCE and reused                              |
        +--------------------------------------------------------------+
                               |
                               v
        +--------------------------------------------------------------+
        | Step 3: get_blink_statistic(...)                              |
        |   Output: blink_stats (blinkAmpRatio, bestMedian, robustStd,  |
        |                     goodRatio, numberGoodBlinks, cutoff, ...) |
        |                                                              |
        |   MATLAB equivalent: line-by-line stats inside extractBlinks.m |
        |   (amp ratio + cutoff ratios computation block)               |
        +--------------------------------------------------------------+
                               |
                               v
        +--------------------------------------------------------------+
        | Step 4: get_good_blink_mask(...)                              |
        |   Output: filtered df (keep only good blinks by z-threshold)  |
        |                                                              |
        |   MATLAB equivalent: getGoodBlinkMask(...)                     |
        |   (called inside extractBlinkProperties.m)                    |
        +--------------------------------------------------------------+
                               |
                               v
        +--------------------------------------------------------------+
        | Step 5: BlinkProperties(...).df                               |
        |   Output: df_out (per-blink durations, pAVR, times, peaks, ..)|
        |                                                              |
        |   MATLAB equivalent: extractBlinkProperties.m per-blink loop  |
        |   (duration*, amp-vel ratios, time shut, peak/interblink)     |
        +--------------------------------------------------------------+
                               |
                               v
        +--------------------------------------------------------------+
        | Step 6: Apply pAVR restriction                                |
        |   df_out = df_out[~(pAVR<threshold & max_value<(bestMedian-std))] |
        |                                                              |
        |   MATLAB equivalent: final restriction block in               |
        |   extractBlinkProperties.m (pMask removal)                    |
        +--------------------------------------------------------------+
                               |
                               v
        +--------------------------------------------------------------+
        | Step 7: Store results per channel                             |
        |   all_data.append(blink_stats)                                |
        |   all_data_info.append({"df": df_out, "ch": channel})         |
        |                                                              |
        |   MATLAB equivalent: signalData struct content across pipeline |
        +--------------------------------------------------------------+
                               |
                               v
        +--------------------------------------------------------------+
        | Step 8: channel_selection(channel_blink_stats, params)        |
        |                                                              |
        |   MATLAB equivalent: post-loop selection in extractBlinks.m   |
        |   - filter by blinkAmpRange                                   |
        |   - filter by minGoodBlinks                                   |
        |   - filter by goodRatioThreshold (+ keepSignals)              |
        |   - select max(numberGoodBlinks)                              |
        +--------------------------------------------------------------+
                               |
                               v
        +--------------------------------------------------------------+
        | Step 9: Final output                                          |
        |   ✓ Selected channel stats + stored df_out (properties)       |
        |   ✓ No recomputation after selection                          |
        |                                                              |
        |   MATLAB contrast: selected signal returns to pop_blinker,    |
        |   then extractBlinkProperties recomputes fitBlinks + props    |
        +--------------------------------------------------------------+
                               |
                               v
        +------------------------------+
        |            End               |
        +------------------------------+
```
