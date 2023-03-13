import json
import time

import numpy
import numpy as np
import pandas as pd
from anytree import *
from flask import Flask, request, jsonify, redirect
from joblib import Parallel, delayed

app = Flask("FIONA")


def read_data(filename):
    return pd.read_csv(filename, dtype=str)


# TODO: Fix here the problem with generalised unique elements
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


def feature_to_split_on(specificity_level, df: numpy.ndarray, name: str):
    """
    :param specificity_level: Specificity level corresponds to the depth of the tree. Based on the depth of the tree
    different kind of splits are done.
    :param df: The dataframe to be split into sub-dataframes.
    :return: The set of the distinct possible options on which the root algorithm will split upon.
    """
    if specificity_level == -2:
        res = [i for n, i in enumerate(df[:, 1]) if i not in df[:, 1][:n]]
        return res
    elif specificity_level == -1:
        length = np.vectorize(len)
        result = set(length(df)[:, 0])
        return result
    else:
        character_split = lambda x: x[:specificity_level + 1]
        word_split = np.vectorize(character_split)
        return set(np.append(word_split(df[:, 0]), str(name) * (len(df[0, 0]) // len(str(name)))))


def tree_grow(column: pd.DataFrame, nDistinctMin=2):
    """
        The tree grow algorithm. It works by adding nodes in the list and popping the head each time in order to split
        it. Initially the tree contains only the root. The split doesn't happen if the unique elements inside the node
        are less than nDistinctMin parameter and the specificity level (i.e. depth+2) is more than 0.
    :param column: An attribute of the database in Pandas Dataframe format upon which to build the tree
    :param nDistinctMin: A pruning parameter which stops the current branch if less than n distinct values in the node
    are found (nDistinctMin doesn't work at a specificity level < 0)
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
                                                   df=current_node.data, name=current_node.name)
        if len(children_identifiers) == 1 and current_node.specificity_level == len(str(current_node.data[0, 0])):
            continue
        surviving_data = current_node.data
        children_identifiers = list(children_identifiers)
        (children_identifiers).sort(key=lambda s: len(str(s)), reverse=True)
        for item in children_identifiers:
            main_data = surviving_data
            if current_node.specificity_level == -2:
                positions = np.nonzero(np.isin(current_node.data[:, 1], item))
                data_for_child = current_node.data[positions]
            elif current_node.specificity_level == -1:
                length = np.vectorize(len)
                positions = np.nonzero(np.isin(length(current_node.data[:, 0]), item))
                data_for_child = current_node.data[positions]

            else:
                positions = [i for i, si in enumerate(main_data[:, 0]) if si.startswith(item)]
                if len(positions) == 0:
                    continue
                temp_data = main_data[positions]
                surviving_data = np.delete(main_data[:], positions, axis=0)
                data_for_child = temp_data

            if current_node.specificity_level < 0:
                node_id = current_node.specificity_level + 1
            else:
                node_id = data_for_child[0, 0][:current_node.specificity_level + 1]
            child = Node(node_id, parent=current_node, data=data_for_child,
                         specificity_level=current_node.specificity_level + 1)
            child_list.append(child)
            node_list.append(child)
        current_node.children = child_list
    return root


def calculate_penalty(specificity_level: int, penalty: int):
    return int(penalty / (specificity_level + 1))


def node_distance(node1: Node, node2: Node, penalty: int):
    sameClass = node1.data[:, 1][0] == node2.data[:, 1][0]
    sameLength = len(node1.data[:, 0][0]) == len(node2.data[:, 0][0])
    if sameClass:
        if sameLength:
            for char_pos in range(len(node1.data[0, 0])):
                if node1.data[0, 0][char_pos] == node2.data[0, 0][char_pos]:
                    continue
                else:
                    break
            return node1.specificity_level + node2.specificity_level - 2 * char_pos
        else:
            # add the specificity levels  +2
            return node1.specificity_level + node2.specificity_level + 2
    else:
        return node1.specificity_level + node2.specificity_level + 4 + calculate_penalty(
            node1.specificity_level, penalty=penalty) + calculate_penalty(node2.specificity_level, penalty=penalty)

def score_function(leaves: tuple):
    """
        Computes a score for each pair of leaves. Score function depends on three quantities:
        (i): The product of elements in each leaf (i.e. masses multiplication)
        (ii): The square of distance of the leaves in the tree.
        (iii): The absolute value of the difference of the elements of the pair of leaves.
        Each quantity is stored in a different matrix in the same position and the tree quantities are multiplied in the
        end creating the score matrix function.
    :param leaves:
    :return: A stack of matrices. The first 3 (indices 0-1-2) have the components of the function, while index 3 has the
            final score.
    """
    # Order is depth,height,width
    max_depth = max([leaf.depth for leaf in leaves])
    matrix = np.empty([4, len(leaves), len(leaves)], dtype='float64')
    matrix.fill(0)

    # TODO: Maybe parallelise it here?
    for i in range(len(leaves)):
        t = time.time()
        for j in range(i, len(leaves)):
            if i == j:
                matrix[0][i][j] = 0
                matrix[1][i][j] = 0
            else:
                masses_multiplication = (leaves[i].data.shape[0] * leaves[j].data.shape[0])
                matrix[0][i][j] = masses_multiplication

                matrix[1][i][j] = (node_distance(leaves[i], leaves[j], max_depth)) ** 2

            masses_difference = abs(leaves[i].data.shape[0] - leaves[j].data.shape[0]) + 1
            matrix[2][i][j] = masses_difference
        print("loops in " + str(time.time() - t))

    outcome = np.divide(matrix[0], matrix[1], out=np.zeros_like(matrix[0]), where=matrix[1] != 0)
    outcome = outcome + outcome.T
    outcome2 = np.divide(outcome, matrix[2], out=np.zeros_like(outcome), where=matrix[2] != 0)
    matrix[3] = outcome2 + outcome2.T
    matrix[2] = matrix[2] + matrix[2].T
    matrix[1] = matrix[1] + matrix[1].T
    matrix[0] = matrix[0] + matrix[0].T
    return matrix


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save("../resources/json_dumps/" + f.filename)
        outlying_elements = process(file=f)
        json_serialised = json.dumps(outlying_elements)
        with open("../resources/json_dumps/" + f.filename + ".json", "w") as outfile:
            outfile.write(json_serialised)
        redirection = redirect("http://localhost:3000/results")
        redirection.headers.add('Access-Control-Allow-Origin', '*')
        return redirection


@app.route("/api/fetch/<string:filename>", methods=['GET'])
def fetch(filename):
    with open("../resources/json_dumps/" + filename + ".json") as f:
        data = json.load(f)
    response = jsonify(data)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


# TODO: Fix here not completely correct check notes
def partial_derivative(odd_case: bool, alpha_list: list, beta_list: list, gamma_list: list):
    if odd_case:
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


def process_attribute(attribute_to_process: str, dataframe: pd.DataFrame):
    column = attribute_to_process
    attribute = process_data(pd.DataFrame(dataframe[column]))
    root = tree_grow(attribute)
    leaves = root.leaves
    # graph.add_nodes_from(leaves)
    # distance_matrix = create_distance_matrix(leaves)
    matrices_packet = score_function(leaves)
    score_matrix = matrices_packet[3]
    medians = np.ma.median(np.ma.masked_invalid(score_matrix, 0), axis=1).data
    median_of_medians = np.median(medians)
    # We need those four lines for the explanation of the results
    # difference = abs(medians - median_of_medians)
    # median_list = np.where(difference == difference.min())[0]
    # mean_absolute_deviation = abs(medians - median_of_medians)
    outlying_elements = {}

    threshold_values = np.linspace(0.0002, 1, 1100)
    previous_values = [-1, -1]
    upper_outlying_indices_dict = {}
    lower_outlying_indices_dict = {}
    for threshold in threshold_values:
        element_dict = {}
        upper_outlying_indices = np.argwhere(medians > (median_of_medians * threshold))
        lower_outlying_indices = np.argwhere(medians < (median_of_medians * threshold))
        if upper_outlying_indices.shape[0] == previous_values[0] and lower_outlying_indices.shape[0] == \
                previous_values[1]:
            continue
        else:
            previous_values[0] = upper_outlying_indices.shape[0]
            previous_values[1] = lower_outlying_indices.shape[0]
            upper_outlying_indices_dict[threshold] = upper_outlying_indices
            lower_outlying_indices_dict[threshold] = lower_outlying_indices
            # for i in upper_outlying_indices:
            #     element_dict[leaves[i[0]].data[:, 0][0]] = leaves[i[0]].data.shape[0]
            for j in lower_outlying_indices:
                element_dict[leaves[j[0]].data[:, 0][0]] = leaves[j[0]].data.shape[0]
            outlying_elements[threshold] = element_dict
    return outlying_elements

    # add here the explanation of the results
    # TODO: fix here and put dict
    # for index in upper_outlying_indices:
    #     median = medians[index]
    #     alpha = matrices_packet[0][:][index]
    #     beta = matrices_packet[1][:][index]
    #     gamma = matrices_packet[2][:][index]
    #     score = score_matrix[index]
    #     dif = abs(score - median)
    #     indices_of_median = np.where(dif == dif.min())[1]
    #
    #     specific_alphas = alpha[0][indices_of_median]
    #     specific_betas = beta[0][indices_of_median]
    #     specific_gammas = gamma[0][indices_of_median]
    #
    #     # Here start doing the partial derivative
    #     # TODO fix the partial derivative
    #     partial_a, partial_b, partial_c = partial_derivative(oddCase=True, alpha_list=specific_alphas,
    #                                                          beta_list=specific_betas,
    #                                                          gamma_list=specific_gammas)
    # print(" ")

    # return jsonify(upper_outlying_indices_dict,lower_outlying_indices_dict)

    # if lower_outlying_indices.shape[0] == 0 and lower_outlying_indices.shape[0] == 0:
    #     print("NOTHING TO REPORT for the column " + column)
    # else:
    #     print("Report outliers for the column " + column)
    #     for i in upper_outlying_indices:
    #         element, count = np.unique(leaves[i[0]].data[:, 0], return_counts=True)
    #         print(str(element) + " appears " + str(count) + " times")
    #     for j in lower_outlying_indices:
    #         element, count = np.unique(leaves[j[0]].data[:, 0], return_counts=True)
    #         print(str(element) + " appears " + str(count) + " times")


def add_outlying_elements_to_attribute(column: str, output: dict, dataframe: pd.DataFrame):
    output[str(column)] = process_attribute(column, dataframe)
    print("Finished " + column)
    return output


def process(file: str, multiprocess_switch):
    # dataframe = read_data("../resources/datasets/datasets_testing_purposes/testing123.csv")
    # dataframe = read_data("../resources/datasets/datasets_testing_purposes/dirty.csv")
    # dataframe = read_data("../resources/datasets/datasets_testing_purposes/10492-1.csv")
    dataframe = read_data("../resources/datasets/datasets_testing_purposes/adult.csv")
    # dataframe = read_data("../resources/json_dumps/" + file.filename)
    output = {}

    if multiprocess_switch:
        output = Parallel(n_jobs=-1)(
            delayed(add_outlying_elements_to_attribute)(column, output, dataframe) for column in dataframe.columns)
    else:
        for column in dataframe.columns:
            column = "fnlwgt"
            output = add_outlying_elements_to_attribute(column, output, dataframe)
    return output


if __name__ == "__main__":
    # app.run(host="0.0.0.0", debug=False)
    t = time.time()
    filename = "../json_dumps/testing123.csv"
    multiprocess = False
    big_dict = process(filename, multiprocess)
    print(time.time() - t)
