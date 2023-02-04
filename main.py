import random

import anytree.util
import networkx as nx
import numpy
import numpy as np
import pandas as pd
from anytree import *
from flask import Flask, request

app = Flask("test")


def read_data(filename):
    return pd.read_csv(filename, dtype=str)


def process_data(dataframe: pd.DataFrame):
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
        result = set(length(df)[:, 0])
        return result
    else:
        gen = np.vectorize(generalise_string, excluded=['specificity_level'])
        return set(gen(df[:, 0], specificity_level))


def tree_grow(column, nDistinctMin=2):
    root = Node("root", children=[], data=np.asarray(column), specificity_level=-2)
    node_list = [root]
    while node_list:
        current_node = node_list.pop(0)
        child_list = []
        if np.unique(current_node.data[:, 0]).shape[0] < nDistinctMin and current_node.specificity_level > 0:
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
    return (node1.depth + node2.depth) - 2 * anytree.util.commonancestors(node1, node2)[-1].depth


def create_distance_matrix(leaves: tuple):
    matrix = np.empty([len(leaves), len(leaves)])
    for first_index in range(len(leaves)):
        for second_index in range(first_index, len(leaves)):
            if first_index == second_index:
                matrix[first_index][second_index] = 0
            else:
                matrix[first_index][second_index] = (node_distance(leaves[first_index], leaves[second_index]))
    matrix = matrix + matrix.T
    return matrix


def score_function(leaves: tuple, distance_matrix: numpy.ndarray):
    # Order is depth,height,width
    matrix = np.empty([3, len(leaves), len(leaves)])
    for i in range(len(distance_matrix)):
        for j in range(i, len(distance_matrix[0])):
            masses_multiplication = leaves[i].data.shape[0] * leaves[j].data.shape[0]
            matrix[0][i][j] = masses_multiplication

            distance_squared = ((distance_matrix[i][j]) ** 2)
            matrix[1][i][j] = distance_squared

            masses_difference = abs(leaves[i].data.shape[0] - leaves[j].data.shape[0]) + 1
            matrix[2][i][j] = masses_difference

    outcome = np.divide(matrix[0], matrix[1], out=np.zeros_like(matrix[0]), where=matrix[1] != 0)
    outcome2 = np.divide(outcome, matrix[2], out=np.zeros_like(outcome), where=matrix[2] != 0)
    outcome2 = outcome2 + outcome2.T
    return outcome2


def calculate_upper_lower_outliers():
    pass


@app.route('/')
def hello():
    return "<p>Hello, World!</p>"


@app.route('/upload_file', methods=['POST'])
def upload_file():
    file = request.files['file']
    print(file)
    return "done"


if __name__ == "__main__":
    # app.run(debug=False)
    dataframe = read_data("resources/datasets/adult.csv")
    graph = nx.Graph()
    for column in dataframe.columns:
        # column="short_name"
        attribute = process_data(pd.DataFrame(dataframe[column]))
        root = tree_grow(attribute)
        leaves = root.leaves
        graph.add_nodes_from(leaves)
        distance_matrix = create_distance_matrix(leaves)
        score_matrix = score_function(leaves, distance_matrix)
        medians = np.ma.median(np.ma.masked_invalid(score_matrix, 0), axis=1).data
        median_of_medians = np.median(medians)
        mean_absolute_deviation = abs(medians - median_of_medians)
        # TODO : Make threshold dynamic
        threshold = 0.4826
        upper_outlying_indices = np.argwhere(medians > (median_of_medians + median_of_medians * threshold))
        lower_outlying_indices = np.argwhere(medians < (median_of_medians - median_of_medians * threshold))
        # add here the explanation of the results
        example_index = upper_outlying_indices[4]
        example_median = medians[example_index]

        print("    ")

        if lower_outlying_indices.shape[0] == 0 and lower_outlying_indices.shape[0] == 0:
            print("NOTHING TO REPORT")
        else:
            print("Report outliers for the column " + column)
            for i in upper_outlying_indices:
                element, count = np.unique(leaves[i.item].data[:, 0], return_counts=True)
                print(str(element) + " appears " + str(count) + " times")
            for j in lower_outlying_indices:
                element, count = np.unique(leaves[j.item].data[:, 0], return_counts=True)
                print(str(element) + " appears " + str(count) + " times")
    print('')
