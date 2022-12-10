import random

import numpy as np
import pandas as pd
from anytree import *


def read_data(filename):
    return pd.read_csv(filename)


def process_data(dataframe):
    dataframe['GeneralisedUniqueElements'] = dataframe.applymap(generalise_string).applymap(find_unique_elements)
    return dataframe


def generalise_string(string):
    for character in string:
        if character.isupper():
            string = string.replace(character, "U")
        if character.islower():
            string = string.replace(character, "l")
        if character.isdigit():
            string = string.replace(character, "d")
        if character.isspace():
            string = string.replace(character, "s")
    return string


def find_unique_elements(generalised_string):
    return set(generalised_string)


def feature_to_split_on(specificity_level, df):
    if specificity_level == -2:
        return np.unique(df[:, 1])
    elif specificity_level == -1:
        length = np.vectorize(len)
        return length(df)[:, 0]
    else:
        print("Split based on specificity_level of the first element")


def tree_grow(column, nmin=6):
    root = Node("root", children=[], data=np.asarray(column), specificity_level=-2)
    node_list = [root]
    while node_list:
        current_node = node_list.pop(0)
        child_list = []
        if len(current_node.data) < nmin:
            continue
        children_identifiers = feature_to_split_on(specificity_level=current_node.specificity_level,
                                                   df=current_node.data)
        for item in children_identifiers:
            if current_node.specificity_level == -2:
                positions = np.nonzero(np.isin(current_node.data[:, 1], item))
            elif current_node.specificity_level == -1:
                length = np.vectorize(len)
                positions = np.nonzero(np.isin(length(current_node.data[:, 0]), item))
            data_for_child = current_node.data[positions[0]]
            child = Node(current_node.name + str(random.random), data=data_for_child,
                         specificity_level=current_node.specificity_level + 1)
            child_list.append(child)
            node_list.append(child)
        current_node.children = child_list
    return root


if __name__ == "__main__":
    dataframe = read_data("testing.csv")
    dataframe = process_data(dataframe)
    tree=tree_grow(dataframe)
    print('123')
