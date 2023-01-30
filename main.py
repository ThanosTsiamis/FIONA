import random

import anytree.util
import numpy
import numpy as np
import pandas as pd
from anytree import *


def read_data(filename):
    return pd.read_csv(filename, dtype=str)


def process_data(dataframe):
    # TODO: Apply numpy speedup here by using np array instead of dataframe
    # dataframe=dataframe.to_numpy()
    # something = np.vectorize(generalise_string)
    # uniquely=np.vectorize(find_unique_elements)
    dataframe = dataframe.fillna('')
    dataframe['GeneralisedUniqueElements'] = dataframe.applymap(generalise_string).applymap(find_unique_elements)
    return dataframe


def generalise_string(string: str, specificity_level=0):
    string = str(string)
    if specificity_level >= 0:
        original_part = list(string[:specificity_level])
        edited_part = list(string[specificity_level:])
    for index in range(len(edited_part)):
        if edited_part[index].isupper():
            edited_part[index] = "U"
        elif edited_part[index].islower():
            edited_part[index] = "l"
        elif edited_part[index].isdigit():
            edited_part[index] = "d"
        elif edited_part[index].isspace():
            edited_part[index] = "w"
        elif not edited_part[index].isalnum():
            edited_part[index] = "s"
    return "".join(original_part + edited_part)


def find_unique_elements(generalised_string):
    return set(generalised_string)


def unique_array_fixed(uniqued_array: numpy.ndarray):
    # TODO: Find why do we have to use that here instead of actual unique
    next_unique = np.unique(uniqued_array)
    while next_unique.size != uniqued_array.size:
        uniqued_array = next_unique
        next_unique = np.unique(next_unique)
    return uniqued_array


def feature_to_split_on(specificity_level, df):
    if specificity_level == -2:
        # TODO : FIX HERE
        res = [i for n, i in enumerate(df[:, 1]) if i not in df[:, 1][:n]]
        return res
    elif specificity_level == -1:
        length = np.vectorize(len)
        return set(length(df)[:, 0])
    else:
        gen = np.vectorize(generalise_string, excluded=['specificity_level'])
        return set(gen(df[:, 0], specificity_level))


def tree_grow(column, nmin=30):
    root = Node("root", children=[], data=np.asarray(column), specificity_level=-2)
    node_list = [root]
    while node_list:
        current_node = node_list.pop(0)
        child_list = []
        if len(current_node.data) < nmin and current_node.specificity_level > 0:
            continue
        children_identifiers = feature_to_split_on(specificity_level=current_node.specificity_level,
                                                   df=current_node.data)
        if len(children_identifiers) == 1 and current_node.specificity_level == len(str(current_node.data[0, 0])):
            continue
        for item in children_identifiers:
            if current_node.specificity_level == -2:
                positions = np.nonzero(np.isin(current_node.data[:, 1], item))
            elif current_node.specificity_level == -1:
                length = np.vectorize(len)
                positions = np.nonzero(np.isin(length(current_node.data[:, 0]), item))
            else:
                gen = np.vectorize(generalise_string, excluded=['specificity_level'])
                positions = np.nonzero(np.isin(gen(current_node.data[:, 0], current_node.specificity_level), item))
            data_for_child = current_node.data[positions[0]]
            child = Node(current_node.name + str(random.random), parent=current_node, data=data_for_child,
                         specificity_level=current_node.specificity_level + 1)
            child_list.append(child)
            node_list.append(child)
        current_node.children = child_list
    return root


def node_distance(node1: Node, node2: Node):
    return node1.depth + node2.depth - 2 * anytree.util.commonancestors(node1, node2)[-1].depth


def create_distance_matrix(leaves: tuple):
    matrix = []
    for first_index in range(len(leaves)):
        row = []
        for second_index in range(len(leaves)):
            row.append(node_distance(leaves[first_index], leaves[second_index]))
        matrix.append(row)
    return np.asarray(matrix)


def create_enforced_siblings_vector(leaves: tuple):
    # what I mean by enforced is that 2 leaves that have each one sibling but the one has more family should be
    # preferred over the other
    matrix: list = []
    for first_index in range(len(leaves)):
        siblings_tuple: tuple = leaves[first_index].siblings
        siblings_population = 0
        for sibling in siblings_tuple:
            siblings_population += len(sibling.data)
        print("123")

    # for node in leaves:
    #     pass


def score_function(leaves: tuple, distance_matrix: numpy.ndarray):
    matrix:list = []
    for i in range(len(distance_matrix)):
        row = []
        for j in range(len(distance_matrix[0])):
            row.append(leaves[i].data.shape[0] *leaves[j].data.shape[0]/(distance_matrix[i][j])**2)
        matrix.append(row)
    return np.asarray(matrix)


if __name__ == "__main__":
    dataframe = read_data("resources/datasets/10492-1.csv")
    for column in dataframe.columns:
        attribute = process_data(pd.DataFrame(dataframe[column]))
        root = tree_grow(attribute)
        leaves = root.leaves
        distance_matrix = create_distance_matrix(leaves)
        score_matrix = score_function(leaves,distance_matrix)
        print("123")
        # create_enforced_siblings_vector(leaves)
    print('123')
