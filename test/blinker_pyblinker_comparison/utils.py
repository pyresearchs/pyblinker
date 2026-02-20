import numpy as np
from pathlib import Path
from scipy.io import loadmat


TEST_ROOT = Path(__file__).resolve().parents[1]
TEST_FILES_DIR = TEST_ROOT / "test_files"


def test_file_path(filename: str) -> Path:
	"""Return a path under ``test/test_files`` independent of current working dir."""
	return TEST_FILES_DIR / filename



def load_matlab_blink_positions(path: str | Path) -> np.ndarray:
	"""Load blinkPositions from a complex MATLAB structure.

	Structure expected: blinks -> signalData(1) -> blinkPositions
	"""
	p = str(path)
	mat = loadmat(p, squeeze_me=True)

	if "blinks" not in mat:
		raise KeyError(f"'blinks' structure not found in {p}. Keys: {list(mat.keys())}")

	blinks = mat["blinks"]

	# Check for signalData
	if "signalData" not in blinks.dtype.names:
		raise KeyError(f"'signalData' not found in blinks structure. Fields: {blinks.dtype.names}")

	signal_data = blinks["signalData"].item() # Get the array of structs

	# Handle the case where signal_data might be a single struct or an array of structs
	if hasattr(signal_data, 'dtype') and signal_data.dtype.names is not None:
		# It's a single struct (likely because of squeeze_me=True on a 1x1 array)
		first_signal = signal_data
	elif isinstance(signal_data, (np.ndarray, list)) and len(signal_data) > 0:
		first_signal = signal_data[0]
	else:
		first_signal = signal_data

	if not hasattr(first_signal, 'dtype') or first_signal.dtype.names is None:
		raise ValueError(f"Could not extract a valid struct from signalData. Type: {type(first_signal)}")

	if "blinkPositions" not in first_signal.dtype.names:
		raise KeyError(f"'blinkPositions' not found in signalData. Fields: {first_signal.dtype.names}")

	arr = first_signal["blinkPositions"]

	# Debug info
	# print(f"DEBUG: arr type={type(arr)}, shape={getattr(arr, 'shape', 'no shape')}")

	# Handle potentially empty or weird shapes
	if not isinstance(arr, np.ndarray) or arr.size == 0:
		return np.empty((2, 0), dtype=np.int64)

	# If it's a scalar array (0-d), it might be containing another array
	if arr.ndim == 0:
		arr = arr.item()

	# normalize shape to (2, N)
	if arr.ndim == 1:
		arr = arr.reshape(2, 1)
	if arr.shape[0] != 2:
		if arr.ndim == 2 and arr.shape[1] == 2:
			arr = arr.T
		else:
			raise ValueError(f"Unexpected blinkPositions shape {arr.shape}; expected (2, N)")

	# MATLAB is 1-based -> Python 0-based
	return arr.astype(np.int64) - 1
