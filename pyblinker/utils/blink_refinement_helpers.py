from collections import defaultdict
from typing import List, Dict, Any

def group_refined_by_epoch(refined: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, int]]]:
    """
    Groups blink entries by their 'epoch_index'.

    Each entry in `refined` contains frame information (start, peak, end) and an epoch index.
    This function organizes all entries into a dictionary keyed by `epoch_index`, with values
    being lists of entries that fall under the same epoch.

    Example:
        Input:
            [
                {'epoch_index': 0, 'refined_start_frame': 100, ...},
                {'epoch_index': 0, 'refined_start_frame': 130, ...},
                {'epoch_index': 1, 'refined_start_frame': 900, ...}
            ]

        Output:
            {
                0: [dict1, dict2],
                1: [dict3]
            }

    Args:
        refined: List of blink entries, each containing at least 'epoch_index',
                 'refined_start_frame', 'refined_peak_frame', 'refined_end_frame'.

    Returns:
        Dictionary mapping each epoch_index to a list of corresponding blink entries.
    """
    grouped: Dict[int, List[Dict[str, int]]] = defaultdict(list)
    for item in refined:
        epoch = item['epoch_index']
        grouped[epoch].append(item)
    return dict(grouped)
