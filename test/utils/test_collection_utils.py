# """Tests for shared iterable, dict, and string utilities."""
# from __future__ import annotations
#
# import unittest
#
# import numpy as np
# import pandas as pd
#
# from pyblinker.utils.dict_utils import (
#     append_to_slot,
#     contains_key,
#     group_by_key,
#     update_dict_list,
# )
# from pyblinker.utils.iter_utils import ensure_float_list, ensure_list, iter_chunks
# from pyblinker.utils.string_utils import safe_literal_eval
#
#
# class TestStringUtils(unittest.TestCase):
#     def test_safe_literal_eval_parses_list(self) -> None:
#         self.assertEqual(safe_literal_eval("[1, 2, 3]"), [1, 2, 3])
#
#     def test_safe_literal_eval_returns_original_on_error(self) -> None:
#         self.assertEqual(safe_literal_eval("not [a valid literal"), "not [a valid literal")
#
#
# class TestIterUtils(unittest.TestCase):
#     def test_ensure_list_handles_various_inputs(self) -> None:
#         self.assertEqual(ensure_list("[1, 2]"), [1, 2])
#         self.assertEqual(ensure_list((1, 2)), [1, 2])
#         self.assertEqual(ensure_list(np.array([1, 2])), [1, 2])
#         self.assertEqual(ensure_list(pd.Series([1, 2])), [1, 2])
#         self.assertEqual(ensure_list(None)[0], None)
#
#     def test_ensure_float_list_handles_nan_and_none(self) -> None:
#         self.assertEqual(ensure_float_list(None), [])
#         self.assertEqual(ensure_float_list(float("nan")), [])
#         self.assertEqual(ensure_float_list("[1, 2.5]"), [1.0, 2.5])
#
#     def test_iter_chunks_splits_iterable(self) -> None:
#         chunks = list(iter_chunks(range(5), 2))
#         self.assertEqual(chunks, [[0, 1], [2, 3], [4]])
#
#
# class TestDictUtils(unittest.TestCase):
#     def test_append_to_slot_behaviour(self) -> None:
#         self.assertEqual(append_to_slot([1], 2), [1, 2])
#         self.assertEqual(append_to_slot(float("nan"), 3), [3])
#         self.assertEqual(append_to_slot(4, 5), [4, 5])
#
#     def test_contains_key_supports_series(self) -> None:
#         series = pd.Series({"foo": 1})
#         mapping = {"foo": 2}
#         self.assertTrue(contains_key(series, "foo"))
#         self.assertTrue(contains_key(mapping, "foo"))
#         self.assertFalse(contains_key(mapping, "bar"))
#
#     def test_group_by_key(self) -> None:
#         grouped = group_by_key(
#             [
#                 {"epoch_index": 0, "value": 1},
#                 {"epoch_index": 0, "value": 2},
#                 {"epoch_index": 1, "value": 3},
#             ],
#             "epoch_index",
#         )
#         self.assertEqual(set(grouped.keys()), {0, 1})
#         self.assertEqual(len(grouped[0]), 2)
#
#     def test_update_dict_list(self) -> None:
#         target: dict[str, list[int]] = {}
#         update_dict_list(target, "key", [1, 2])
#         update_dict_list(target, "key", [3])
#         self.assertEqual(target["key"], [1, 2, 3])
#
#
# if __name__ == "__main__":
#     unittest.main()
