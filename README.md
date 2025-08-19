# pyblinker
`pyblinker` is a Python package primarily intended for automated detection of eye-blink artifacts in diverse biosignals, 
including electroencephalography (EEG), electrooculography (EOG), and Eye Aspect Ratio (EAR). It addresses the limitations of 
existing MATLAB-based tools like [BLINKER](https://github.com/VisLab/EEG-Blinks), 
offering a Python-based, open-source solution for researchers analyzing various types of physiological data.

A key benefit of `pyblinker` is its enhanced functionality and broader applicability compared to `BLINKER`.  
By expanding input compatibility beyond EEG/EOG to include EAR, and by integrating seamlessly with the MNE-Python ecosystem, 
`pyblinker` provides a more versatile and accessible tool. This facilitates efficient and reproducible biosignal analysis 
workflows across neuroscience and related fields, ultimately streamlining research and improving data quality.

# Features
`pyblinker` provide a wide variety of tools to use including:

* Extraction of eye blinks from EEG, EOG, and EAR data.
* Compatibility with both continuous and epoch-based datasets.
* Support for single-channel and multi-channel signals.
* Computation of blink characteristics such as amplitude, duration, and peak-to-peak interval.
* Per-epoch feature extraction, including EAR metrics, energy complexity, frequency-domain measures, blink morphology, and waveform features.
* Generation of blink time series for visualization and validation.


# Contributing
Contributions are welcome (more than welcome!). Contributions can be feature requests, improved documentation, bug reports, 
code improvements, new code, etc. Anything will be appreciated. Note: this package adheres to the same contribution standards as MNE.

# Acknowledgements

* RPB wishes to thank the Ministry of Higher Education Malaysia for their financial support via the FRGS-EC 026-2024

* The original [BLINKER](https://github.com/VisLab/EEG-Blinks) algorithms have been ported into the ``pyblinker/blinker`` directory of this repository. These modules retain the
legacy MATLAB logic so that results remain comparable when migrating existing
workflows.

        > K. Kleifges, N. Bigdely-Shamlo, S. E. Kerick, K. A. Robbins (2017). 
        > BLINKER: Automated Extraction of Ocular Indices from EEG Enabling Large-Scale Analysis. 
        > Frontiers in Neuroscience*, 11, Article 12. 
        > https://doi.org/10.3389/fnins.2017.00012
        

* This package is built on top of many other great packages. If you use `pyblinker` you should also acknowledge these packages.
    > [MNE-Python](https://mne.tools/dev/overview/cite.html)


### 📦 Installation

To install directly from GitHub using `pip`, run:

```bash
pip install git+https://github.com/pyresearchs/pyblinker.git
```

# Documentation
For a detailed guide on `pyblinker`, see the [tutorials](https://github.com/pyresearchs/pyblinker/tree/main/tutorial).
