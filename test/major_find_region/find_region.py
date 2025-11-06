import logging

import mne
from pyblinker.blinker.pyblinker import BlinkDetector

logging.basicConfig(level=logging.INFO)


raw_file=r'C:\Users\balan\IdeaProjects\pyblinker\test\test_files\seg_annotated_raw.fif'


raw = mne.io.read_raw_fif(raw_file, preload=True)
raw.pick_types(eeg=True)
# drange=[f'E{X}' for X in [8,9,10,14,15,16,17]]
# to_drop_ch = list(set(raw.ch_names) - set(drange))
# raw = raw.drop_channels(to_drop_ch)

annot, ch, number_good_blinks, df, fig_data, ch_selected = BlinkDetector(raw, visualize=False, annot_label=None,
                                                                         filter_low=0.5, filter_high=30.0, resample_rate=100,
                                                                         n_jobs=2,use_multiprocessing=True).get_blink()
#98 E8
print(f"Total number of detected eye close events: {number_good_blinks}, or {len(df)}")
raw.set_annotations(annot)
raw.plot(block=True, title=f'Eye close based on channel {ch}')
