'''
Find the blinks in the EOG channel of the sample dataset and create annotations for them.
This code uses MNE-Python to find EOG events, create annotations for blink events, and visualize the results.
It demonstrates how to process EEG data, identify eye blinks, and visualize the results using MNE-Python.

'''

import os
import mne
import matplotlib
matplotlib.use('TkAgg')


sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_filt-0-40_raw.fif')

raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)
raw.pick_types(meg=False, eeg=False, eog=True, ecg=False)

events_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                           'sample_audvis_filt-0-40_raw-eve.fif')


eog_events = mne.preprocessing.find_eog_events(raw)
onsets = eog_events[:, 0] / raw.info['sfreq'] - 0.25
durations = [0.5] * len(eog_events)
descriptions = ['bad blink'] * len(eog_events)
blink_annot = mne.Annotations(onsets, durations, descriptions,
                              orig_time=raw.info['meas_date'])
raw.set_annotations(blink_annot)
epochs = mne.make_fixed_length_epochs(raw, duration=10, preload=False)
eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True)
raw.plot(events=eog_events,block=True)