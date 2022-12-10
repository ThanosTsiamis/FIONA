import numpy as np
import pandas as pd
from anytree import *


def read_data(filename):
    return pd.read_csv(filename)


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
    return (set(generalised_string))


def feature_to_split_on(specifity_level, df):
    if specifity_level == -2:
        foobar = df.applymap(generalise_string).applymap(find_unique_elements)
        return np.unique(np.asarray(foobar))
    elif specifity_level == -1:
        print("Split based on Length")
    else:
        print("Split based on specifity_level of the first element")


def tree_grow(column, nmin=6):
    root = Node("root", children=None, data=column, specifity_level=-2)
    node_list = [root]
    while node_list:
        current_node = node_list.pop(0)
        first_value = current_node.data.iloc[0].values[0]
        generalised_string = generalise_string(first_value)
        generalised_set = set(generalised_string)
        feature_to_split_on(specifity_level=current_node.specifity_level, df=current_node.data)
        print("123")


if __name__ == "__main__":
    dataframe = read_data("testing.csv")
    tree_grow(dataframe)
