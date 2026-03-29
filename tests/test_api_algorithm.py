import unittest

from tests.module_loader import load_module


algorithm = load_module("test_algorithm_module", "api/algorithm.py")


class AlgorithmHelpersTest(unittest.TestCase):
    def test_generalise_string_maps_character_classes(self):
        self.assertEqual(algorithm.generalise_string("Ab 3!"), "Ulwds")

    def test_generalise_string_preserves_prefix_for_specificity(self):
        self.assertEqual(algorithm.generalise_string("AbZ9", specificity_level=2), "AbUd")

    def test_replace_repeated_chars_collapses_runs(self):
        self.assertEqual(algorithm.replace_repeated_chars("UUUUddsU"), "U+d+sU")

    def test_find_difference_index_handles_same_and_different_lengths(self):
        self.assertEqual(algorithm.find_difference_index("alpha", "alpha"), -1)
        self.assertEqual(algorithm.find_difference_index("alpha", "alpHa"), 3)
        self.assertEqual(algorithm.find_difference_index("abc", "ab"), 2)

    def test_compare_dicts_checks_nested_content(self):
        left = {"outer": {"value": 1}, "name": "same"}
        right = {"name": "same", "outer": {"value": 1}}
        other = {"outer": {"value": 2}, "name": "same"}

        self.assertTrue(algorithm.compare_dicts(left, right))
        self.assertFalse(algorithm.compare_dicts(left, other))

    def test_merge_dictionaries_sums_overlapping_nested_values(self):
        merged = algorithm.merge_dictionaries(
            {"A": {"x": 1, "y": 2}, "B": {"z": 5}},
            {"A": {"y": 3, "z": 4}, "C": {"w": 6}},
        )

        self.assertEqual(
            merged,
            {
                "A": {"x": 1, "y": 5, "z": 4},
                "B": {"z": 5},
                "C": {"w": 6},
            },
        )

    def test_convert_to_percentage_recurses_through_nested_dicts(self):
        converted = algorithm.convert_to_percentage(
            {"A": {"x": 2, "nested": {"y": 1}}, "label": "keep"},
            4,
        )

        self.assertEqual(
            converted,
            {"A": {"x": 0.5, "nested": {"y": 0.25}}, "label": "keep"},
        )

    def test_build_algorithm_config_uses_explicit_request_settings(self):
        config = algorithm.build_algorithm_config(
            manual_override_ndistinct=9,
            manual_override_long_column=10,
            manual_override_large_file_threshold=12,
            regex_transformation_only=True,
            generalised_transformation_only=False,
        )

        self.assertEqual(config.ndistinct_manual_setting, 9)
        self.assertEqual(config.long_column_limit, 10)
        self.assertEqual(config.large_file_threshold, 12)
        self.assertTrue(config.regex_only)
        self.assertFalse(config.generalised_only)


if __name__ == "__main__":
    unittest.main()
