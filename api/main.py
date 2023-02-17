import random

import anytree.util
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
    """

    :param specificity_level:
    :param df:
    :return:
    """
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


def tree_grow(column: pd.DataFrame, nDistinctMin=2):
    """
    :param column: An attribute of the database in Pandas Dataframe format upon which to build the tree
    :param nDistinctMin: A pruning parameter which stops the current branch if less than n distinct values in the node
    are found
    :return: the root of the tree
    """
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
            child = Node(str(random.random), parent=current_node, data=data_for_child,
                         specificity_level=current_node.specificity_level + 1)
            child_list.append(child)
            node_list.append(child)
        current_node.children = child_list
    return root


def node_distance(node1: Node, node2: Node):
    return (node1.depth + node2.depth) - 2 * anytree.util.commonancestors(node1, node2)[-1].depth


def create_distance_matrix(leaves: tuple):
    matrix = np.empty([len(leaves), len(leaves)])
    matrix.fill(0)
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
    matrix.fill(0)
    for i in range(len(distance_matrix)):
        for j in range(i, len(distance_matrix[0])):
            if i == j:
                matrix[0][i][j] = 0
            else:
                masses_multiplication = (leaves[i].data.shape[0] * leaves[j].data.shape[0])
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


@app.route('/')
def hello():
    return "<p>Hello, World!</p>"


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        return "<p> OKOKOK </p>"


# TODO: Fix here not completely correct check notes
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


if __name__ == "__main__":
    # app.run(host="0.0.0.0", debug=False)
    dataframe = read_data("../resources/datasets/datasets_testing_purposes/10492-1.csv")
    for column in dataframe.columns:
        # column = "Account Name"
        attribute = process_data(pd.DataFrame(dataframe[column]))
        root = tree_grow(attribute)
        leaves = root.leaves
        # graph.add_nodes_from(leaves)
        distance_matrix = create_distance_matrix(leaves)
        matrices_packet = score_function(leaves, distance_matrix)
        score_matrix = matrices_packet[3]
        medians = np.ma.median(np.ma.masked_invalid(score_matrix, 0), axis=1).data
        median_of_medians = np.median(medians)
        difference = abs(medians - median_of_medians)
        median_list = np.where(difference == difference.min())[0]
        mean_absolute_deviation = abs(medians - median_of_medians)
        # TODO : Make threshold dynamic
        threshold_values = np.linspace(0.20, 1, 120)
        previous_values = [-1, -1]
        upper_outlying_indices_dict = {}
        lower_outlying_indices_dict = {}
        for threshold in threshold_values:
            upper_outlying_indices = np.argwhere(medians > (median_of_medians + median_of_medians * threshold))
            lower_outlying_indices = np.argwhere(medians < (median_of_medians - median_of_medians * threshold))
            if upper_outlying_indices.shape[0] == previous_values[0] and lower_outlying_indices.shape[0] == \
                    previous_values[1]:
                continue
            else:
                previous_values[0] = upper_outlying_indices.shape[0]
                previous_values[1] = lower_outlying_indices.shape[0]
                upper_outlying_indices_dict[threshold] = upper_outlying_indices
                lower_outlying_indices_dict[threshold] = lower_outlying_indices

        # add here the explanation of the results

        for index in upper_outlying_indices:
            median = medians[index]
            alpha = matrices_packet[0][:][index]
            beta = matrices_packet[1][:][index]
            gamma = matrices_packet[2][:][index]
            score = score_matrix[index]
            dif = abs(score - median)
            indices_of_median = np.where(dif == dif.min())[1]

            specific_alphas = alpha[0][indices_of_median]
            specific_betas = beta[0][indices_of_median]
            specific_gammas = gamma[0][indices_of_median]

            # Here start doing the partial derivative
            # TODO fix the partial derivative
            partial_a, partial_b, partial_c = partial_derivative(oddCase=True, alpha_list=specific_alphas,
                                                                 beta_list=specific_betas,
                                                                 gamma_list=specific_gammas)
        print(" ")

        if lower_outlying_indices.shape[0] == 0 and lower_outlying_indices.shape[0] == 0:
            print("NOTHING TO REPORT for the column " + column)
        else:
            print("Report outliers for the column " + column)
            for i in upper_outlying_indices:
                element, count = np.unique(leaves[i[0]].data[:, 0], return_counts=True)
                print(str(element) + " appears " + str(count) + " times")
            for j in lower_outlying_indices:
                element, count = np.unique(leaves[j[0]].data[:, 0], return_counts=True)
                print(str(element) + " appears " + str(count) + " times")
    print('')
