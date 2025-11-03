# Test EDF provenance and rationale

This repository includes a small EDF file under `test/test_files/` that is generated from the public MNE-Python "sample" dataset. We save it in EDF format to enable 1:1 comparisons with the MATLAB version of the BLINKER package, which commonly uses EDF inputs.

- Source: MNE "sample" dataset (public, bundled via `mne.datasets.sample.data_path()`).
- Conversion: We read `sample_audvis_filt-0-40_raw.fif` and export to EDF using MNE's export utilities.
- Location: `test/test_files/mne_sample_audvis_raw.edf` (created on-demand).

Recreate locally:

1. Ensure dependencies are installed (see `requirements.txt`): `mne` is required.
2. Run the helper to fetch the dataset and create the EDF:

```bat
python -m test.data_setup --make-edf
```

The first run will download the MNE sample dataset to your MNE data directory, then export the EDF into `test/test_files`. Subsequent runs will reuse the generated EDF unless `--overwrite` is passed.

Why EDF?

- EDF is widely supported by MATLAB toolboxes and the legacy BLINKER workflow.
- Using EDF avoids format-specific differences when comparing against MATLAB.

If you need to trace the exact code path, see `test/data_setup.py` and the `ensure_mne_sample_edf` function.
note also, the file is big and some data is not related, therefore, we do the selection as follows:

raw.pick_types(eeg=True)
raw.filter(0.5, 20.5, fir_design='firwin')
raw.resample(100)

    drange=[f'EEG 00{X}' for X in [1,2,3,5,8]]
    # drange=[f'EEG 00{X}' for X in range(10)]
    to_drop_ch = list(set(raw.ch_names) - set(drange))
    if to_drop_ch:
        raw = raw.drop_channels(to_drop_ch)

