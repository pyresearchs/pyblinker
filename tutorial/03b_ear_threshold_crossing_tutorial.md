# EAR Threshold Crossings and Slope Features

This tutorial demonstrates how to detect threshold crossings in an Eye Aspect Ratio (EAR) time series, identify the minimum inside a blink, and compute closing/opening slope features. The approach mirrors the geometry shown in the demo script from the prompt: first downward crossing → minimum → first upward crossing after the minimum.

## Definitions

- **Left threshold crossing** (`t_L`): first downward crossing of `EAR = θ` (values go above → below) inside the search region.
- **Minimum** (`t_min`, `y_min`): smallest EAR after `t_L`.
- **Right threshold crossing** (`t_R`): first upward crossing of `EAR = θ` after the minimum (values go below → above).
- **Closing slope**: \( m_{close} = \dfrac{y_{min} - \theta}{t_{min} - t_L} \) (negative).
- **Opening slope**: \( m_{open} = \dfrac{\theta - y_{min}}{t_R - t_{min}} \) (positive).

Crossings are bracketed by sign changes relative to `θ` and refined with **linear interpolation**. If a crossing segment is flat (`y0 == y1`), the **plateau policy** uses the midpoint by default (configurable to `"left"` or `"right"`).

## Mock EAR blink

```python
import numpy as np
import matplotlib.pyplot as plt

from pyblinker.fitutils.ear_crossing import (
    compute_threshold_slopes,
    find_threshold_crossing_triplet,
)

# Synthetic EAR with a Gaussian-shaped blink dip
fs = 250
t = np.arange(0, 2.0, 1 / fs)
rng = np.random.default_rng(7)
baseline = 0.30
noise = 0.004 * rng.standard_normal(len(t))
center = 1.0
sigma = 0.07
depth = 0.19
dip = depth * np.exp(-0.5 * ((t - center) / sigma) ** 2)
ear = baseline - dip + noise
theta = 0.185

# Crossing + slope computation
triplet = find_threshold_crossing_triplet(
    ear,
    theta=theta,
    t=t,
    window=(0, len(ear) - 1),
    max_expansion=int(0.05 * fs),  # allow up to 50 ms outward search
    expansion_step=max(1, int(0.01 * fs)),
    plateau_policy="midpoint",
)
closing_slope, opening_slope = compute_threshold_slopes(triplet, theta)

print("Left crossing:", triplet.left.time)
print("Minimum:", triplet.minimum_time, triplet.minimum_value)
print("Right crossing:", triplet.right.time)
print("Closing slope:", closing_slope)
print("Opening slope:", opening_slope)
```

## Optional plotting

```python
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(t, ear, s=18, color="#1f77b4", alpha=0.75, label="EAR samples")
ax.plot(t, ear, lw=2.5, color="#1f77b4", alpha=0.35, label="EAR (line)")

ax.axhline(theta, color="crimson", lw=2, ls="--", label=f"Threshold θ = {theta:.3f}")
ax.scatter(
    [triplet.left.time, triplet.minimum_time, triplet.right.time],
    [theta, triplet.minimum_value, theta],
    s=120,
    color=["crimson", "black", "crimson"],
    alpha=0.5,
    zorder=5,
)

ax.plot(
    [triplet.left.time, triplet.minimum_time],
    [theta, triplet.minimum_value],
    color="black",
    lw=2.5,
    alpha=0.4,
    label=f"Closing slope = {closing_slope:.2f}",
)
ax.plot(
    [triplet.minimum_time, triplet.right.time],
    [triplet.minimum_value, theta],
    color="green",
    lw=2.5,
    alpha=0.4,
    label=f"Opening slope = {opening_slope:.2f}",
)

# Vertical guides
ax.axvline(triplet.left.time, color="crimson", alpha=0.4)
ax.axvline(triplet.right.time, color="crimson", alpha=0.4)
ax.axvline(triplet.minimum_time, color="black", alpha=0.3)

# Annotations (matching the demo snippet)
ax.annotate(
    "Left threshold crossing",
    xy=(triplet.left.time, theta),
    xytext=(triplet.left.time - 0.35, theta + 0.06),
    arrowprops=dict(arrowstyle="->", lw=2, color="crimson"),
    fontsize=11,
    color="crimson",
)

ax.annotate(
    "Minimum EAR",
    xy=(triplet.minimum_time, triplet.minimum_value),
    xytext=(triplet.minimum_time + 0.10, triplet.minimum_value - 0.08),
    arrowprops=dict(arrowstyle="->", lw=2),
    fontsize=11,
)

ax.annotate(
    "Right threshold crossing",
    xy=(triplet.right.time, theta),
    xytext=(triplet.right.time + 0.05, theta + 0.06),
    arrowprops=dict(arrowstyle="->", lw=2, color="crimson"),
    fontsize=11,
    color="crimson",
)

# Slope labels
ax.text(
    (triplet.left.time + triplet.minimum_time) / 2,
    (theta + triplet.minimum_value) / 2 + 0.01,
    f"closing slope = {closing_slope:.2f} EAR/s",
    bbox=dict(boxstyle="round", fc="white", ec="black"),
    fontsize=11,
)

ax.text(
    (triplet.minimum_time + triplet.right.time) / 2,
    (theta + triplet.minimum_value) / 2 - 0.04,
    f"opening slope = {opening_slope:.2f} EAR/s",
    bbox=dict(boxstyle="round", fc="white", ec="green"),
    fontsize=11,
    color="green",
)

ax.set_title("EAR threshold crossings with slopes")
ax.set_xlabel("Time (s)")
ax.set_ylabel("EAR")
ax.set_ylim(0.05, 0.36)
ax.legend(loc="lower left")
ax.grid(alpha=0.25)
fig.tight_layout()
```

## Edge-case behavior

- **Window expansion**: If the blink window omits a crossing, set `max_expansion`/`expansion_step` to allow outward search. The result flags `found_by="expanded"` when expansion was required.
- **No crossings**: A `ThresholdCrossingError` is raised when no consistent triple is found (even after expansion), allowing callers to fall back deterministically.
- **Multiple noisy crossings**: The algorithm always uses the **first** downward crossing within the window (or expanded window), the **minimum after that**, and then the **first** upward crossing after the minimum to enforce event consistency.
- **Flat segments at θ**: When a bracketed segment is flat, the interpolation uses the midpoint (or the configured policy) so slope calculations remain finite.
- **Division-by-zero safety**: If the minimum time equals a crossing time, slopes return `nan` rather than raising.

You can adapt the search `window` to each blink (e.g., refined onset/offset samples) and reuse `find_threshold_crossing_triplet` anywhere an EAR-like threshold crossing must be detected.
