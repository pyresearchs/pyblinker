#
# import scipy.io
# import numpy as np
# import mne
# import scipy.io
# import numpy as np
# import mne
# def load_mat_to_mne(mat_path: str, sfreq_default: float = 256.0) -> mne.io.BaseRaw:
#     def print_structure(data, indent=0):
#         if isinstance(data, dict):
#             for k, v in data.items():
#                 if k.startswith('__'):
#                     continue
#                 print("  " * indent + f"Key: {k}, Type: {type(v)}")
#                 print_structure(v, indent + 1)
#         elif isinstance(data, np.ndarray):
#             print("  " * indent + f"ndarray with dtype: {data.dtype}, shape: {data.shape}")
#
#     try:
#         mat = scipy.io.loadmat(mat_path, simplify_cells=True)
#     except Exception as e:
#         print(f"Error loading .mat file with simplify_cells=True: {e}")
#         try:
#             mat = scipy.io.loadmat(mat_path)
#             print("Successfully loaded with simplify_cells=False. Structure:")
#             print_structure(mat)
#         except Exception as e2:
#             print(f"Failed to load even with simplify_cells=False: {e2}")
#         raise ValueError("Could not load .mat file properly.")
#
#     def find_numeric_arrays(data, path=''):
#         found = {}
#         if isinstance(data, dict):
#             for k, v in data.items():
#                 if k.startswith('__'):
#                     continue
#                 new_path = f"{path}/{k}" if path else k
#                 found.update(find_numeric_arrays(v, new_path))
#         elif isinstance(data, np.ndarray) and data.dtype.kind in 'fiu' and data.size > 0:
#             found[path] = data
#         return found
#
#     candidate_arrays = find_numeric_arrays(mat)
#     if not candidate_arrays:
#         print("No numeric arrays found in the .mat file after recursive search.")
#         print("MAT-file structure:"); print_structure(mat)
#         raise ValueError("No numeric arrays found in the .mat file.")
#
#     # pick the largest numeric array
#     best_key, best_arr = max(candidate_arrays.items(), key=lambda kv: kv[1].size)
#     arr = np.asarray(best_arr)
#
#     # ---- shape to (n_channels, n_times) -------------------------------------
#     if arr.ndim == 1:
#         arr = arr[np.newaxis, :]
#     elif arr.ndim == 3:
#         # collapse first two dims if they look like (epochs, channels, time)
#         if arr.shape[-1] >= max(arr.shape[0], arr.shape[1]):  # time is last
#             arr = arr.reshape(-1, arr.shape[-1])
#         else:
#             arr = arr[0]
#     elif arr.ndim > 3:
#         raise ValueError(f"Array {best_key} has unsupported ndim={arr.ndim}")
#
#     # decide orientation by taking the longer dimension as time
#     if arr.shape[0] > arr.shape[1]:
#         arr = arr.T  # (time, ch) -> (ch, time)
#
#     data = arr.astype(np.float64, copy=False)
#
#     # replace inf with nan, then nan -> 0 (MNE requires finite)
#     data[~np.isfinite(data)] = np.nan
#     if np.isnan(data).any():
#         # zero-fill; you could also interpolate if desired
#         data = np.nan_to_num(data, nan=0.0)
#
#     # ---- sampling frequency --------------------------------------------------
#     sfreq = None
#     for k in ['sfreq', 'srate', 'fs', 'sampling_rate', 'Fs','sampFreq']:
#         if k in mat:
#             val = mat[k]
#             try:
#                 sfreq = float(np.asarray(val).ravel()[0])
#                 break
#             except Exception:
#                 pass
#     if sfreq is None:
#         sfreq = float(sfreq_default)
#
#     # ---- channel names -------------------------------------------------------
#     ch_names = None
#     for k in ['ch_names', 'labels', 'chan_names', 'channels', 'labels_names','chnames']:
#         if k in mat:
#             v = mat[k]
#             try:
#                 names = [str(x) for x in np.atleast_1d(v).ravel()]
#                 if len(names) == data.shape[0]:
#                     ch_names = names
#             except Exception:
#                 pass
#             break
#     if ch_names is None:
#         ch_names = [f"CH{i+1}" for i in range(data.shape[0])]
#
#     # ---- UNIT NORMALIZATION (critical) --------------------------------------
#     # MNE expects Volts. Many .mat files store microvolts or millivolts.
#     scale = 1.0
#
#     # 1) metadata hints
#     meta_str = " ".join(map(str, mat.keys())).lower()
#     if any(tok in meta_str for tok in ['uv', 'microv', 'microvolt', 'micro_volt']):
#         scale = 1e6
#     elif any(tok in meta_str for tok in ['mv', 'milliv', 'millivolt', 'milli_volt']):
#         scale = 1e3
#
#     # 2) heuristic fallback based on magnitude (99th percentile of |data|)
#     if scale == 1.0:
#         p99 = float(np.percentile(np.abs(data), 99))
#         # typical EEG in volts: ~1e-6 .. 1e-4; if values are O(10..1000), assume µV
#         if p99 > 1.0:              # e.g., 100 µV stored as 100
#             scale = 1e6           # µV -> V
#         elif 1e-2 < p99 <= 1.0:    # e.g., 0.05 (50 mV) -> likely mV
#             scale = 1e3           # mV -> V
#         # else already in volts
#
#     if scale != 1.0:
#         print(f"[load_mat_to_mne] Detected non-Volt data. Dividing by {scale:g} to convert to Volts.")
#         data = data / scale
#
#     # sanity print
#     p99_v = float(np.percentile(np.abs(data), 99))
#     print(f"[load_mat_to_mne] 99th percentile amplitude ≈ {p99_v:.3e} V ({p99_v*1e6:.1f} µV)")
#
#     # ---- build Raw -----------------------------------------------------------
#     info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
#     raw = mne.io.RawArray(data, info)  # data now in Volts
#
#     return raw



import scipy.io
import numpy as np
import mne
import scipy.io
import numpy as np
import mne

def print_structure(data, indent=0):
    if isinstance(data, dict):
        for k, v in data.items():
            if k.startswith('__'):
                continue
            print("  " * indent + f"Key: {k}, Type: {type(v)}")
            print_structure(v, indent + 1)
    elif isinstance(data, np.ndarray):
        print("  " * indent + f"ndarray with dtype: {data.dtype}, shape: {data.shape}")

def find_numeric_arrays(data, path=''):
        found = {}
        if isinstance(data, dict):
            for k, v in data.items():
                if k.startswith('__'):
                    continue
                new_path = f"{path}/{k}" if path else k
                found.update(find_numeric_arrays(v, new_path))
        elif isinstance(data, np.ndarray) and data.dtype.kind in 'fiu' and data.size > 0:
            found[path] = data
        return found
def load_mat_to_mne(mat_path: str, sfreq_default: float = 256.0) -> mne.io.BaseRaw:



    mat = scipy.io.loadmat(mat_path, simplify_cells=True)
    best_arr=mat['o']['data']
    arr = np.asarray(best_arr)


    # decide orientation by taking the longer dimension as time
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T  # (time, ch) -> (ch, time)

    data = arr.astype(np.float64, copy=False)

    # replace inf with nan, then nan -> 0 (MNE requires finite)
    data[~np.isfinite(data)] = np.nan
    if np.isnan(data).any():
        # zero-fill; you could also interpolate if desired
        data = np.nan_to_num(data, nan=0.0)

    # ---- sampling frequency --------------------------------------------------
    sfreq = mat['o']['sampFreq']
    ch_names = [f"CH{i+1}" for i in range(data.shape[0])]

    # ---- UNIT NORMALIZATION (critical) --------------------------------------
    # MNE expects Volts. Many .mat files store microvolts or millivolts.
    # 2) heuristic fallback based on magnitude (99th percentile of |data|)
    # if scale == 1.0:
    p99 = float(np.percentile(np.abs(data), 99))
    # typical EEG in volts: ~1e-6 .. 1e-4; if values are O(10..1000), assume µV
    if p99 > 1.0:              # e.g., 100 µV stored as 100
        scale = 1e6           # µV -> V
    elif 1e-2 < p99 <= 1.0:    # e.g., 0.05 (50 mV) -> likely mV
        scale = 1e3           # mV -> V
    # else already in volts

    if scale != 1.0:
        print(f"[load_mat_to_mne] Detected non-Volt data. Dividing by {scale:g} to convert to Volts.")
        data = data / scale



    # ---- build Raw -----------------------------------------------------------
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info)  # data now in Volts

    return raw
