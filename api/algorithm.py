import gc

import numpy as np
import pandas as pd
from anytree import Node
from joblib import Parallel, delayed


def read_data(filename):
    file_extension = filename.split(".")[-1]
    if file_extension == "csv":
        return pd.read_csv(filename, dtype=str)
    elif file_extension == "xlsx":
        return pd.read_excel(filename)
    elif file_extension == "json":
        return pd.read_json(filename)
    else:
        return print("DATABASE NOT YET SUPPORTED")


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


def feature_to_split_on(specificity_level, df: np.ndarray, name: str):
    """
    This function determines the kind of split that will be applied on the data base at hand. This is mostly dependent
    on the specificity_level (i.e. the depth of the tree). However, for depths greater than 2, an extra criterion is
    added which check for repeating values. This is done by multiplying the non abstract part of the name of the node
    as much as necessary in order to produce repeating patterns.
    :param specificity_level: Specificity level corresponds to the depth of the tree. Based on the depth of the tree
    different kind of splits are done.
    :param df: The dataframe to be split into sub-dataframes.
    :return: The set of the distinct possible options on which the root algorithm will split upon.
    """
    if specificity_level == -2:
        unique_sets = []
        hash_table = {}
        for s in df[:, 1]:
            s_set = set(s)
            hash_val = hash(frozenset(s_set))
            if hash_val not in hash_table:
                hash_table[hash_val] = s_set
                unique_sets.append(s_set)
        return unique_sets
    elif specificity_level == -1:
        length = np.vectorize(len)
        result = set(length(df)[:, 0])
        return result
    else:
        character_split = lambda x: x[:specificity_level + 1]
        word_split = np.vectorize(character_split)
        if specificity_level == 0:
            return set(word_split(df[:, 0]))
        else:
            return set(np.append(word_split(df[:, 0]),
                                 str(name)[:specificity_level] * (
                                         len(df[0, 0]) // len(str(name[:specificity_level]))
                                 )
                                 )
                       )


def calculate_machine_limit():
    try:
        for i in range(1000, 200000, 1000):
            in_memory_variable = np.empty([4, i, i], dtype='float32')
            gc.collect()
    except:
        gc.collect()
        return i - 1000


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
    # pruning_preparations = False
    while node_list:
        current_node = node_list.pop(0)
        child_list = []
        if np.unique(current_node.data[:, 0]).shape[0] < nDistinctMin and current_node.specificity_level > 0:
            continue
        children_identifiers = feature_to_split_on(specificity_level=current_node.specificity_level,
                                                   df=current_node.data, name=current_node.name)
        if current_node.specificity_level == -2:
            limit_of_machine = calculate_machine_limit()
            if (np.unique(current_node.data[:, 0]).size == current_node.data[:, 0].size) or (
                    np.unique(current_node.data[:, 0]).size > limit_of_machine):
                if len(children_identifiers) == 1:
                    if children_identifiers[0] == {'d'} or children_identifiers[0] == {'d', 's'}:
                        # TODO:Fix the latter case e.g. sddssdds
                        continue
        #         else:
        #             fifteen_percent_of_limit = int(limit_of_machine * 0.15)
        #             children_occ_dict = {}
        #             children_percentage_dict = {}
        #             pruning_preparations = True
        #             for kid in children_identifiers:
        #                 children_occ_dict[frozenset(kid)] = np.unique(
        #                     current_node.data[current_node.data[:, 1] == kid][:, 0]).size
        #             sorted_dict = {k: v for k, v in sorted(children_occ_dict.items(), key=lambda item: item[1])}
        #             for kid in sorted_dict.keys():
        #                 if children_occ_dict[kid] < fifteen_percent_of_limit:
        #                     print(kid)
        #                 else:
        #                     break
        #                     #Pare ayto to kid kai arxise na moirazeis me pososta
        #             summation = sum(children_occ_dict.values())
        #             for val in children_occ_dict:
        #                 children_percentage_dict[val] = children_occ_dict[val] / summation
        #
        # if pruning_preparations == True and current_node.specificity_level == -1:
        if len(children_identifiers) == 1 and current_node.specificity_level == len(str(current_node.data[0, 0])):
            continue
        surviving_data = current_node.data
        children_identifiers = list(children_identifiers)
        children_identifiers.sort(key=lambda s: len(str(s)), reverse=True)
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
                node_id = generalise_string(data_for_child[0, 0], current_node.specificity_level + 1)
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
    specificity_sum = node1.specificity_level + node2.specificity_level
    if sameClass:
        if sameLength:
            str1 = node1.data[0, 0]
            str2 = node2.data[0, 0]
            if str1 and str2:
                char_pos = np.argmax(np.array(list(str1)) != np.array(list(str2)))
                return specificity_sum - 2 * char_pos
            else:
                return specificity_sum + 2
        else:
            return specificity_sum + 2
    else:
        penalty_sum = calculate_penalty(node1.specificity_level, penalty=penalty) + calculate_penalty(
            node2.specificity_level, penalty=penalty)
        return specificity_sum + 4 + penalty_sum


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
    try:
        gc.collect()
        matrix = np.empty([4, len(leaves), len(leaves)], dtype='float32')
    except:
        gc.collect()
        matrix = np.empty([4, len(leaves), len(leaves)], dtype='float32')
    matrix.fill(0)

    # TODO: Maybe parallelise it here?
    for i in range(len(leaves)):
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

    outcome = np.divide(matrix[0], matrix[1], out=np.zeros_like(matrix[0]), where=matrix[1] != 0)
    outcome = outcome + outcome.T
    outcome2 = np.divide(outcome, matrix[2], out=np.zeros_like(outcome), where=matrix[2] != 0)
    matrix[3] = outcome2 + outcome2.T
    matrix[2] = matrix[2] + matrix[2].T
    matrix[1] = matrix[1] + matrix[1].T
    matrix[0] = matrix[0] + matrix[0].T
    return matrix


def process_attribute(attribute_to_process: str, dataframe: pd.DataFrame):
    column = attribute_to_process
    attribute = process_data(pd.DataFrame(dataframe[column]))
    machine_limit = calculate_machine_limit()
    root = tree_grow(attribute)
    ndistinct = 2
    while True:
        leaves = root.leaves
        if len(leaves) < machine_limit:
            break
        else:
            str_x = str(ndistinct)
            if len(str_x) == 1:
                ndistinct += 1
            elif str_x[-1] == '9':
                ndistinct = int('1' + '0' * len(str_x))
            else:
                ndistinct += 1
            root = tree_grow(attribute, nDistinctMin=ndistinct)
    print(attribute)
    matrices_packet = score_function(leaves)
    score_matrix = matrices_packet[3]
    medians = np.ma.median(np.ma.masked_invalid(score_matrix, 0), axis=1).data
    output_dictionary = {}
    outlying_elements = {}
    pattern_elements = {}

    threshold_values = np.linspace(1, 50, 2500)
    previous_values = [-1, -1]
    pattern_indices_dict = {}
    lower_outlying_indices_dict = {}
    for threshold in threshold_values:
        element_dict_outliers = {}
        element_dict_patterns = {}
        outlier_threshold = np.percentile(medians, threshold)
        lower_outlying_indices = np.argwhere(medians < outlier_threshold)
        pattern_indices = np.argwhere(medians > outlier_threshold)

        if pattern_indices.shape[0] == previous_values[0] and lower_outlying_indices.shape[0] == previous_values[1]:
            continue
        if lower_outlying_indices.shape[0] == previous_values[1]:
            continue
        else:
            previous_values[0] = pattern_indices.shape[0]
            previous_values[1] = lower_outlying_indices.shape[0]
            pattern_indices_dict[threshold] = pattern_indices
            lower_outlying_indices_dict[threshold] = lower_outlying_indices
            for i in pattern_indices:
                uniques, counts = (np.unique(leaves[i[0]].data[:, 0], return_counts=True))
                element_dict_patterns[leaves[i[0]].name] = dict(np.asarray((uniques, counts)).T)
            for j in lower_outlying_indices:
                uniques, counts = (np.unique(leaves[j[0]].data[:, 0], return_counts=True))
                element_dict_outliers[leaves[j[0]].name] = dict(np.asarray((uniques, counts)).T)
            outlying_elements[threshold] = element_dict_outliers
            pattern_elements[threshold] = element_dict_patterns
    output_dictionary["outliers"] = outlying_elements
    output_dictionary["patterns"] = pattern_elements
    return output_dictionary


def compare_dicts(dict1, dict2):
    # Check if both dictionaries have the same keys
    if set(dict1.keys()) != set(dict2.keys()):
        return False

    # Iterate through each key in the dictionary
    for key in dict1.keys():
        # Check if the value for the key is another dictionary
        if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            # Recursively check if the nested dictionaries are the same
            if not compare_dicts(dict1[key], dict2[key]):
                return False
        else:
            # Check if the values for the key are the same
            if dict1[key] != dict2[key]:
                return False

    # If all key-value pairs match, return True
    return True


def add_outlying_elements_to_attribute(column: str, dataframe: pd.DataFrame):
    col_outliers_and_patterns = process_attribute(column, dataframe)
    lexicon = {column: {}}
    has_previous_threshold_dict = False
    previous_threshold_dict_value = -1
    marked_for_clearance = []
    for threshold_level in col_outliers_and_patterns['outliers'].keys():
        lexicon[column][threshold_level] = {}
        inner_dicts = col_outliers_and_patterns['patterns'][threshold_level]
        pattern_set = {generalise_string(key) for inner_dict in inner_dicts.values() for key in inner_dict.keys()}
        for outlier_rep in col_outliers_and_patterns['outliers'][threshold_level].keys():
            first_outliers_element = list(col_outliers_and_patterns['outliers'][threshold_level][outlier_rep])[0]
            generalised_pattern = generalise_string(first_outliers_element)
            if generalised_pattern in pattern_set:
                continue
            else:
                inner_dict = lexicon[column][threshold_level].get(generalised_pattern, {})
                inner_dict.update(col_outliers_and_patterns['outliers'][threshold_level][outlier_rep])
                lexicon[column][threshold_level][generalised_pattern] = inner_dict
        if has_previous_threshold_dict:
            current_dict = lexicon[column][threshold_level]
            previous_dict = lexicon[column][previous_threshold_dict_value]
            if compare_dicts(current_dict, previous_dict):
                marked_for_clearance.append(threshold_level)
        has_previous_threshold_dict = True
        previous_threshold_dict_value = threshold_level
    for threshold_level in marked_for_clearance:
        del lexicon[column][threshold_level]

    print("Finished " + column)
    return lexicon


def process_column(column, dataframe):
    return add_outlying_elements_to_attribute(column, dataframe)


def process(file: str, multiprocess_switch):
    # dataframe = read_data("../resources/datasets/datasets_testing_purposes/testing123.csv")
    # dataframe = read_data("resources/datasets/datasets_testing_purposes/dirty.csv")
    # dataframe = read_data("resources/json_dumps/flightsDirty.csv")
    # dataframe = read_data("../resources/json_dumps/School_Learning_Modalities__2020-2021.csv")
    # dataframe = read_data("../resources/json_dumps/pima-indians-diabetes.csv")
    # dataframe = read_data("../resources/datasets/datasets_testing_purposes/10492-1.csv")
    # dataframe = read_data("../resources/datasets/datasets_testing_purposes/16834-1.csv")
    # dataframe = read_data("../resources/datasets/datasets_testing_purposes/adult.csv")
    # dataframe = read_data("resources/datasets/datasets_testing_purposes/hospital/HospitalDirty.csv")
    # dataframe = read_data("resources/datasets/datasets_testing_purposes/tax/taxClean.csv")
    dataframe = read_data("resources/json_dumps/" + file.filename)
    output = {}

    if multiprocess_switch == "True":
        output = {}
        results = Parallel(n_jobs=-1)(
            delayed(process_column)(column, dataframe) for column in dataframe.columns
        )

        for res in results:
            output.update(res)
    else:
        for column in dataframe.columns:
            if column == "salary":
                output.update(add_outlying_elements_to_attribute(column, dataframe))
    return output
