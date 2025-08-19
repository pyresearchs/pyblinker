import logging

import mne
import numpy as np

logger = logging.getLogger(__name__)


def compute_ear_calibration(
    open_or_annotated: "mne.io.BaseRaw | mne.Epochs | np.ndarray",
    closed: "mne.io.BaseRaw | mne.Epochs | np.ndarray | None" = None,
    *,
    ch_name: str,
    open_label: str = "eyes_open_calib",
    closed_label: str = "eyes_closed_calib",
    open_pct: float = 0.75,
    closed_pct: float = 0.25,
) -> dict:
    """Compute robust EAR levels for eyes-open and eyes-closed states.

    This helper supports three calibration strategies:

    1. **Annotated recording** – provide a single :class:`mne.io.BaseRaw` or
       :class:`mne.Epochs` object that contains annotations marking eyes-open
       (``open_label``) and eyes-closed (``closed_label``) calibration spans.
       In this mode ``closed`` must be ``None`` and the annotations are used to
       gather calibration samples.
    2. **Separate recordings** – provide eyes-open data via ``open_or_annotated``
       and eyes-closed data via ``closed``. Each may be a Raw, Epochs or a 1-D
       ``numpy.ndarray``. When using MNE objects, ``ch_name`` selects the EAR
       channel.
    3. **Manual thresholds** – if the open/closed EAR levels are known a priori
       or are being tuned externally, bypass this function and pass the levels
       directly to the detector as ``{"open": open_level, "closed": closed_level}``
       through the ``calibration`` entry in the parameter dictionary.

    ### 📈 Pipeline Flowchart

    ```plaintext
               +-------------+
               | Input data  |
               +-------------+
                     |
                     v
          +-----------------------+
          |   Annotated data?     |
          +-----------------------+
            | yes             | no
            v                 v
 +--------------------+   +------------------------------+
 | Collect spans by   |   | Use separate open/closed     |
 | labels             |   | recordings                   |
 +--------------------+   +------------------------------+
            \\               /
             v             v
          +-----------------------+
          | Compute percentiles   |
          +-----------------------+
                     |
                     v
          +-----------------------+
          | {"open", "closed"}    |
          +-----------------------+
    ```

    Parameters
    ----------
    open_or_annotated : mne.io.BaseRaw | mne.Epochs | numpy.ndarray
        Eyes-open data or an annotated recording depending on the selected mode.
    closed : mne.io.BaseRaw | mne.Epochs | numpy.ndarray | None
        Eyes-closed data when using the separate recordings mode. Leave as
        ``None`` when annotations are embedded in ``open_or_annotated``.
    ch_name : str
        Name of the EAR channel for MNE objects. Ignored for ``numpy`` arrays.
    open_label : str
        Annotation label marking eyes-open calibration segments.
    closed_label : str
        Annotation label marking eyes-closed calibration segments.
    open_pct : float
        Percentile used to estimate the open-eye EAR level.
    closed_pct : float
        Percentile used to estimate the closed-eye EAR level.

    Returns
    -------
    dict
        ``{"open": float, "closed": float}`` with the estimated EAR levels.

    Raises
    ------
    ValueError
        If insufficient calibration data are provided.
    """
    logger.info("Computing EAR calibration")

    def _to_array(obj: "mne.io.BaseRaw | mne.Epochs | np.ndarray") -> np.ndarray:
        if isinstance(obj, np.ndarray):
            return np.asarray(obj, float).ravel()
        if isinstance(obj, mne.Epochs):
            obj = obj.copy().load_data().to_raw()
        return obj.copy().pick(ch_name).get_data()[0]

    if closed is None:
        raw = open_or_annotated
        if isinstance(raw, np.ndarray):
            raise TypeError(
                "When 'closed' is None, 'open_or_annotated' must be Raw or Epochs with annotations",
            )
        if isinstance(raw, mne.Epochs):
            raw = raw.copy().load_data().to_raw()
        x = raw.copy().pick(ch_name).get_data()[0]

        def _collect(label: str) -> np.ndarray:
            spans = []
            for onset, dur, desc in zip(
                raw.annotations.onset, raw.annotations.duration, raw.annotations.description
            ):
                if desc == label:
                    s0 = int((onset - raw.first_time) * raw.info["sfreq"])
                    s1 = s0 + int(dur * raw.info["sfreq"])
                    s0 = max(0, s0)
                    s1 = min(len(x), s1)
                    if s1 > s0:
                        spans.append(x[s0:s1])
            return np.concatenate(spans) if spans else np.array([])

        xo = _collect(open_label)
        xc = _collect(closed_label)
    else:
        xo = _to_array(open_or_annotated)
        xc = _to_array(closed)

    logger.debug("Collected %d open and %d closed samples", xo.size, xc.size)
    if xo.size < 100 or xc.size < 100:
        raise ValueError("Not enough EAR calibration data (eyes open/closed).")

    open_level = float(np.nanpercentile(xo, open_pct * 100))
    closed_level = float(np.nanpercentile(xc, closed_pct * 100))

    logger.info("Computed EAR calibration open=%.3f closed=%.3f", open_level, closed_level)
    return {"open": open_level, "closed": closed_level}
