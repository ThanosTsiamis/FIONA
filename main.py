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
    matrix = np.empty([4, len(leaves), len(leaves)])
    for i in range(len(distance_matrix)):
        for j in range(i, len(distance_matrix[0])):
            if i == j:
                matrix[0][i][j] = 0
            else:
                masses_multiplication = leaves[i].data.shape[0] * leaves[j].data.shape[0]
                matrix[0][i][j] = masses_multiplication

        distance_squared = ((distance_matrix[i][j]) ** 2)
        matrix[1][i][j] = distance_squared

        masses_difference = abs(leaves[i].data.shape[0] - leaves[j].data.shape[0]) + 1
        matrix[2][i][j] = masses_difference

        outcome = np.divide(matrix[0], matrix[1], out=np.zeros_like(matrix[0]), where=matrix[1] != 0)
        outcome2 = np.divide(outcome, matrix[2], out=np.zeros_like(outcome), where=matrix[2] != 0)
        matrix[3] = outcome2 + outcome2.T
        matrix[2] = matrix[2] + matrix[2].T
        matrix[1] = matrix[1] + matrix[1].T
        matrix[0] = matrix[0] + matrix[0].T
        return matrix


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


def partial_derivative(oddCase: bool, alpha_list: list, beta_list: list, gamma_list: list):
    if oddCase:
        alpha_partial_derivative = (1 / beta_list[0]) * (1 / gamma_list[0])
        beta_partial_derivative = -alpha_list[0] / ((beta_list[0] ** 2) * gamma_list[0])
        gamma_partial_derivative = -alpha_list[0] / ((gamma_list[0] ** 2) * beta_list[0])
    else:
        alpha_partial_derivative = 0.5 * ((1 / (beta_list[0] * gamma_list[0])) + (1 / (beta_list[1] * gamma_list[1])))
        beta_partial_derivative = -0.5 * ((alpha_list[0] / (gamma_list[0] * (beta_list[0] ** 2))) + (
                alpha_list[1] / (gamma_list[1] * (beta_list[1] ** 2))))
        gamma_partial_derivative = -0.5 * ((alpha_list[0] / (beta_list[0] * (gamma_list[0] ** 2))) + (
                alpha_list[1] / (beta_list[1] * (gamma_list[1] ** 2))))
    return alpha_partial_derivative, beta_partial_derivative, gamma_partial_derivative


def get_index_of_median(odd_number_of_elements: bool, input_list: np.ndarray, median):
    if odd_number_of_elements:
        # the median is one of the elements
        return np.argwhere(input_list == median)
    else:
        # find the element closest to the median and take its symmetric based on the median
        # consider the distance of the first element to be the smallest
        min_dif = abs(input_list - median)
        bottom_index = np.argmin(min_dif)
        top_index = np.argwhere(input_list == (min_dif + input_list[bottom_index]))
        return bottom_index, top_index


if __name__ == "__main__":
    # app.run(debug=False)
    dataframe = read_data("resources/datasets/testing.csv")
    graph = nx.Graph()
    for column in dataframe.columns:
        # column="short_name"
        attribute = process_data(pd.DataFrame(dataframe[column]))
        root = tree_grow(attribute)
        leaves = root.leaves
        graph.add_nodes_from(leaves)
        distance_matrix = create_distance_matrix(leaves)
        matrices_packet = score_function(leaves, distance_matrix)
        score_matrix = matrices_packet[3]
        medians = np.ma.median(np.ma.masked_invalid(score_matrix, 0), axis=1).data
        # Median of medians may not be a value of the vector but it may be in between.
        median_of_medians = np.median(medians)
        mean_absolute_deviation = abs(medians - median_of_medians)
        # TODO : Make threshold dynamic
        threshold = 0.4826
        upper_outlying_indices = np.argwhere(medians > (median_of_medians + median_of_medians * threshold))
        lower_outlying_indices = np.argwhere(medians < (median_of_medians - median_of_medians * threshold))
        # add here the explanation of the results
        example_index = upper_outlying_indices[0]
        example_median = medians[example_index]
        # For now assume that the count is odd such that the median is one of them
        # TODO: Assume the second case later

        alpha = matrices_packet[0][:][example_index]
        beta = matrices_packet[1][:][example_index]
        gamma = matrices_packet[2][:][example_index]
        # HERE CHANGE FOR THE SECOND CASE IF IT IS CLOSER
        target_median_index = np.argwhere((alpha * (1 / beta) * (1 / gamma)) == example_median)[0][1]
        specific_alpha = alpha[0][target_median_index]
        specific_beta = beta[0][target_median_index]
        specific_gamma = gamma[0][target_median_index]

        # Here start doing the partial derivative

        partial_a, partial_b, partial_c = partial_derivative(oddCase=True, alpha_list=[specific_alpha],
                                                             beta_list=[specific_beta],
                                                             gamma_list=[specific_gamma])
        print("123")

        index_of_median_of_medians = np.argwhere(median_of_medians == medians)[0][0]

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
