# """Tests for the :func:`create_annotation` utility."""
#
# import logging
# import unittest
#
# import mne
# import pandas as pd
#
# from pyblinker.utils import create_annotation
#
# logger = logging.getLogger(__name__)
#
#
# class TestCreateAnnotation(unittest.TestCase):
#     """Validate annotation creation and input checking."""
#
#     def test_valid_dataframe(self) -> None:
#         """A valid DataFrame should yield an annotation object."""
#         df = pd.DataFrame({"start_blink": [0, 30], "end_blink": [15, 45]})
#         annot = create_annotation(df, 30.0, "blink")
#         self.assertIsInstance(annot, mne.Annotations)
#         self.assertEqual(annot.onset.tolist(), [0.0, 1.0])
#         self.assertEqual(annot.duration.tolist(), [0.5, 0.5])
#         self.assertEqual(annot.description.tolist(), ["blink", "blink"])
#
#     def test_non_dataframe_input_raises(self) -> None:
#         """Non-DataFrame input should raise a ``TypeError``."""
#         with self.assertRaises(TypeError):
#             create_annotation([], 30.0, "blink")
#
#     def test_missing_columns_raises(self) -> None:
#         """Missing required columns should raise a ``ValueError``."""
#         df = pd.DataFrame({"start_blink": [0], "foo": [1]})
#         with self.assertRaises(ValueError):
#             create_annotation(df, 30.0, "blink")
#
#     def test_invalid_sfreq_raises(self) -> None:
#         """Non-positive sampling frequency should raise a ``ValueError``."""
#         df = pd.DataFrame({"start_blink": [0], "end_blink": [1]})
#         with self.assertRaises(ValueError):
#             create_annotation(df, 0.0, "blink")
#
#     def test_invalid_label_raises(self) -> None:
#         """Non-string label should raise a ``TypeError``."""
#         df = pd.DataFrame({"start_blink": [0], "end_blink": [1]})
#         with self.assertRaises(TypeError):
#             create_annotation(df, 30.0, 1)  # type: ignore[arg-type]
#
#
# if __name__ == "__main__":  # pragma: no cover - convenience for developers
#     logging.basicConfig(level=logging.INFO)
#     unittest.main()
#
