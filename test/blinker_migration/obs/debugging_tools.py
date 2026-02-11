from scipy.io import loadmat


def load_matlab_data(input_path=None, output_path=None):
    """
    Load input and output .mat files.

    Args:
        input_path (str): Path to the input .mat file.
        output_path (str): Path to the output .mat file.

    Returns:
        tuple: Input candidate_signal and ground truth output candidate_signal.
    """
    if input_path is not None:
        input_data = loadmat(input_path,    squeeze_me=True,
                             simplify_cells=True,
                             struct_as_record=False)
    else:
        input_data = None

    if output_path is not None:
        output_data = loadmat(output_path,    squeeze_me=True,
                              simplify_cells=True,
                              struct_as_record=False)
    else:
        output_data = None
    return input_data, output_data