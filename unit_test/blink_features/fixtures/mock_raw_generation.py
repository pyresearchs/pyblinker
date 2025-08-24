import numpy as np
import mne


def generate_mock_raw(
    sfreq: float = 50.0,
    epoch_len: float = 10.0,
    n_epochs: int = 3,
) -> mne.io.RawArray:
    """Create a mock Raw object with blink annotations for testing.

    Parameters
    ----------
    sfreq : float
        Sampling frequency of the signal.
    epoch_len : float
        Length of each epoch in seconds.
    n_epochs : int
        Number of epochs to simulate.

    Returns
    -------
    mne.io.RawArray
        Raw signal with blink annotations.
    """
    n_samples = int(sfreq * epoch_len * n_epochs)
    rng = np.random.default_rng(0)
    data = rng.normal(scale=1e-6, size=(1, n_samples))
    info = mne.create_info(["EOG"], sfreq, ["misc"])
    raw = mne.io.RawArray(data, info, verbose=False)
    onsets = np.array([2.0, 5.0, 12.0, 18.0, 22.0])
    durations = np.repeat(0.1, len(onsets))
    raw.set_annotations(mne.Annotations(onsets, durations, ["blink"] * len(onsets)))
    return raw
