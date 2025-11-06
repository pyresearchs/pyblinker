import scipy.io
import numpy as np
import mne
import scipy.io
import numpy as np
import mne

from pyblinker.utils.mat_edf import load_mat_to_mne

if __name__ == "__main__":
    mat_path = r"C:\Users\balan\IdeaProjects\pyblinker\CLA-SubjectJ-170510-3St-LRHand-Inter.mat"
    # mat_path=r"C:\Users\balan\IdeaProjects\blinker_pyblinker_validation\CLA-SubjectJ-170510-3St-LRHand-Inter.mat"
    # or the link https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100295/mat_data/s01.mat
    # dpath=r"s01.mat"
    # mat_path=r"C:\Users\balan\IdeaProjects\pyblinker\S001R01.edf"
    # raw=mne.io.read_raw_edf(mat_path,preload=True)
    # sfreq=float(200.0)
    raw = load_mat_to_mne(mat_path,sfreq_default=sfreq)
    # print(raw)
    # raw.plot(n_channels=min(32, len(raw.ch_names)), scalings='auto', show=True,block=True)

    sfreq=raw.info['sfreq']
    from pyblinker.blinker.pyblinker import BlinkDetector
    drange=[f'CH{X}' for X in range (4)]
    # # # drange=['CH4']
    to_drop_ch = list(set(raw.ch_names) - set(drange))
    raw = raw.drop_channels(to_drop_ch)
    #
    annot, ch, number_good_blinks, blink_details, fig_data, ch_selected = BlinkDetector(raw, visualize=False, annot_label=None,
                                                                                        filter_low=0.5, filter_high=30.0, resample_rate=sfreq,
                                                                                        n_jobs=2,use_multiprocessing=True).get_blink()

    # from pyblinker.viz.report_no_fs import make_blink_report
    #
    # out_html = make_blink_report(
    #     fig_data=fig_data,
    #     ch=ch,
    #     number_good_blinks=number_good_blinks,
    #     ch_selected=ch_selected,
    #     blink_details=blink_details,   # or None if you don't want the table
    #     title=f"Blink Report — channel {ch}",
    #     out_path="blink_report.html",
    #     overwrite=True,
    # )
    # print(f"Report written to: {out_html}")

    raw.set_annotations(annot)
    raw.plot(
        block=True,
        title=f"Eye close based on channel {ch}",
        scalings=10e-6,  # show ±10 µV for all channels
    )