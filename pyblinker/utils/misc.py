import mne
import pandas as pd

def create_annotation(sblink, sfreq, label):


    if not isinstance(sblink, pd.DataFrame):
        raise ValueError('No appropriate channel. sorry. Try to use large channel selection')


    onset = (sblink['start_blink'] / sfreq).tolist()
    duration= (sblink['end_blink'] - sblink['start_blink']) / sfreq
    des_s = [label] * len(onset)

    annot = mne.Annotations(onset=onset,  # in seconds
                            duration=duration,  # in seconds, too
                            description=des_s)

    return annot