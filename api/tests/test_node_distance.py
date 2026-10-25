import numpy as np
from anytree import Node

from algorithm import node_distance


def make_node(value, char_class, specificity_level):
    data = np.empty((1, 2), dtype=object)
    data[0, 0] = value
    data[0, 1] = char_class
    return Node("leaf", data=data, specificity_level=specificity_level)


def test_node_distance_is_zero_when_first_characters_differ_at_specificity_zero():
    # specificity_sum - 2 * char_pos = 0 - 0 = 0. score_function then does 1/distance.
    left = make_node("A", {"U"}, 0)
    right = make_node("B", {"U"}, 0)
    assert node_distance(left, right, penalty=4) == 0


def test_node_distance_is_zero_when_difference_index_matches_specificity():
    left = make_node("ABC", {"U"}, 1)
    right = make_node("ADC", {"U"}, 1)
    assert node_distance(left, right, penalty=4) == 0


def test_node_distance_same_class_same_length_uses_first_difference():
    left = make_node("ABC", {"U"}, 3)
    right = make_node("AXC", {"U"}, 3)
    assert node_distance(left, right, penalty=4) == 6 - 2 * 1


def test_node_distance_different_length_adds_two():
    left = make_node("AA", {"U"}, 1)
    right = make_node("AAA", {"U"}, 1)
    assert node_distance(left, right, penalty=4) == 4


def test_node_distance_different_class_includes_penalty():
    left = make_node("A", {"U"}, 1)
    right = make_node("1", {"d"}, 1)
    # specificity_sum + 4 + int(4/2) + int(4/2) = 2 + 4 + 2 + 2
    assert node_distance(left, right, penalty=4) == 10


def test_node_distance_empty_strings_add_two():
    left = make_node("", set(), 1)
    right = make_node("", set(), 1)
    assert node_distance(left, right, penalty=4) == 4
