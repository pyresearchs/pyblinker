import mne
import numpy as np
from pathlib import Path

def compare_fif_files():
    # Define file paths
    file_paths = {
        "without_annot": Path("test/test_files/ear_eog_without_annotation_raw.fif"),
        "with_annot": Path("test/test_files/ear_eog_raw.fif"),
        "manual_annot": Path("manual_annotation_feature_calculation_data/ear_eog.fif"),
    }
    
    target_channel = "EEG-E8"
    
    # Load data
    raws = {}
    print(f"Loading files to compare channel '{target_channel}'...")
    for key, path in file_paths.items():
        if not path.exists():
            print(f"Error: File not found: {path}")
            return
        try:
            # Verbose=False to reduce clutter
            raws[key] = mne.io.read_raw_fif(path, preload=True, verbose=False)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return

    # 1. Compare Sampling Rates
    sfreqs = {k: raw.info['sfreq'] for k, raw in raws.items()}
    ref_sfreq = sfreqs["without_annot"]
    sfreq_match = all(sf == ref_sfreq for sf in sfreqs.values())
    
    print("\n--- comparison Report ---")
    print(f"Sampling Rates Match: {sfreq_match}")
    for k, sf in sfreqs.items():
        print(f"  {k}: {sf} Hz")
        
    if not sfreq_match:
        print("Stopping comparison due to sampling rate mismatch.")
        return

    # 2. Extract Data for Target Channel
    data_arrays = {}
    for key, raw in raws.items():
        if target_channel not in raw.ch_names:
            print(f"Error: Channel {target_channel} not found in {key}")
            return
        # get_data returns (n_channels, n_times), we take the first (and only) channel
        data_arrays[key] = raw.get_data(picks=[target_channel])[0]

    # 3. Compare Time Series Data
    # We compare against "without_annot" as the reference
    ref_key = "without_annot"
    ref_data = data_arrays[ref_key]
    
    print(f"\nComparing time series against '{ref_key}'...")
    
    is_same_subject = True
    
    for key, data in data_arrays.items():
        if key == ref_key:
            continue
            
        print(f"\nComparing '{key}' vs '{ref_key}':")
        
        # Length check
        if len(data) != len(ref_data):
            print(f"  Length Mismatch! {len(data)} vs {len(ref_data)}")
            is_same_subject = False
            continue
        else:
            print(f"  Lengths match: {len(data)} samples")

        # Value check (exact or near-exact equality)
        # Using allclose for floating point comparisons
        if np.allclose(data, ref_data, atol=1e-10):
            print("  Data Identical: YES")
        else:
            diff = np.abs(data - ref_data)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            print(f"  Data Identical: NO")
            print(f"  Max difference: {max_diff:.6e}")
            print(f"  Mean difference: {mean_diff:.6e}")
            # If correlation is high, it might be the same subject but processed differently
            corr = np.corrcoef(data, ref_data)[0, 1]
            print(f"  Correlation: {corr:.6f}")
            
            if corr < 0.99:
                is_same_subject = False

    print("\n--- Conclusion ---")
    if is_same_subject:
        print("Based on channel EEG-E8, these files appear to be recordings from the SAME subject.")
    else:
        print("These files contain DIFFERENT data for channel EEG-E8 (or different processing).")

if __name__ == "__main__":
    compare_fif_files()
