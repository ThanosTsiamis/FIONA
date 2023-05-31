import gc
import logging

import joblib
import numpy as np
import pandas as pd
from anytree import Node
from joblib import Parallel, delayed

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def read_data(filename):
    file_extension = filename.split(".")[-1]
    if file_extension == "csv":
        return pd.read_csv(filename, dtype=str, encoding='utf-8-sig')
    elif file_extension == "xlsx":
        return pd.read_excel(filename)
    elif file_extension == "json":
        return pd.read_json(filename)
    else:
        raise ValueError("Unsupported file format. Only CSV, XLSX, and JSON formats are supported.")


def process_data(dataframe: pd.DataFrame):
    dataframe = dataframe.fillna('')
    dataframe['GeneralisedUniqueElements'] = dataframe.applymap(generalise_string).applymap(find_unique_elements)
    return dataframe


def generalise_string(string: str, specificity_level=0):
    """
    Convert a string to its generalised equivalent string given a specificity level.
    :param string: a string about to be converted to its generalised equivalent
    :param specificity_level: the current specificity level of the string
    :return: the generalised equivalent of the string
    """
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
    """
    Find the maximum size of an empty NumPy matrix that can be created without causing a memory overflow.
    :return: the maximum size of the matrix.
    """
    try:
        for i in range(1000, 200000, 1000):
            in_memory_variable = np.empty([i, i], dtype='float64')
            gc.collect()
    except MemoryError:
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
    long_column_limit = 36
    limit = calculate_machine_limit()
    while node_list:
        current_node = node_list.pop(0)
        child_list = []
        uniques = np.unique(current_node.data[:, 0])
        if uniques.shape[0] < nDistinctMin and current_node.specificity_level > 0:
            continue
        children_identifiers = feature_to_split_on(specificity_level=current_node.specificity_level,
                                                   df=current_node.data, name=current_node.name)
        if current_node.specificity_level == -2:
            if uniques.size > int(0.95 * current_node.data[:, 0].size):
                if len(children_identifiers) == 1:
                    if children_identifiers[0] == {'d'} or children_identifiers[0] == {'d', 's'}:
                        # TODO:Fix the latter case e.g. sddssdds
                        continue
                # if len(children_identifiers) > 7:
                #     # TODO:THINK OF ANOTHER CUSTOM LIMIT. OR HOW TO DO IT EFFICIENTLY
                #     custom_limit = 600
                #     fifteen_percent_of_limit = int(0.15 * custom_limit)
                #     eightyfive_percent_of_limit = int(0.85 * custom_limit)
                #     children_occ_dict = {}
                #     new_dict = {}
                #     appearances = {}
                #     # pruning_preparations = True
                #     for kid in children_identifiers:
                #         appearance = np.unique(current_node.data[current_node.data[:, 1] == kid][:, 0])
                #         appearances[frozenset(kid)] = appearance
                #         children_occ_dict[frozenset(kid)] = appearance.size
                #     sorted_values = sorted(children_occ_dict.values())
                #     for value in sorted_values:
                #         key = next((k for k, v in children_occ_dict.items() if v == value), None)
                #         if value <= fifteen_percent_of_limit:
                #             new_dict[key] = value  # add the key-value pair to the new dictionary
                #             del children_occ_dict[key]
                #             fifteen_percent_of_limit -= value
                #         else:
                #             new_dict[key] = fifteen_percent_of_limit
                #             children_occ_dict[key] = value - fifteen_percent_of_limit
                #             # fifteen_percent_of_limit=0
                #             break
                #     total_sum = sum(children_occ_dict.values())
                #     remaining_occ_percentage = {}
                #     for key in children_occ_dict:
                #         remaining_occ_percentage[key] = children_occ_dict[key] / total_sum
                #     minimum_85_and_total_sum = min(eightyfive_percent_of_limit, total_sum)
                #     for key in remaining_occ_percentage:
                #         if key in new_dict:
                #             new_dict[key] += int(remaining_occ_percentage[key] * minimum_85_and_total_sum)
                #         else:
                #             new_dict[key] = int(remaining_occ_percentage[key] * minimum_85_and_total_sum)
                #     pile_of_data = np.empty((0, 2))
                #     for key in new_dict:
                #         elements_to_be_kept = appearances[key][:new_dict[key]]
                #         mask = np.where(np.isin(current_node.data[:, 0], elements_to_be_kept))[0]
                #         pile_of_data = np.vstack((pile_of_data, current_node.data[mask]))
                #     current_node.data = pile_of_data
        # if pruning_preparations == True and current_node.specificity_level == -1:
        if current_node.specificity_level == 0 and len(current_node.data[0, 0]) > long_column_limit:
            continue
        if len(children_identifiers) == 1 and current_node.specificity_level == len(str(current_node.data[0, 0])):
            continue
        surviving_data = current_node.data
        children_identifiers = list(children_identifiers)
        children_identifiers.sort(key=lambda s: len(str(s)), reverse=True)
        if current_node.specificity_level == -1:
            children_identifiers.sort()
        breaking_flag = False
        for item in children_identifiers:
            main_data = surviving_data
            if current_node.specificity_level == -2:
                positions = np.nonzero(np.isin(current_node.data[:, 1], item))
                data_for_child = current_node.data[positions]
            elif current_node.specificity_level == -1:
                length = np.vectorize(len)
                if item <= long_column_limit:
                    positions = np.nonzero(np.isin(length(current_node.data[:, 0]), item))
                    data_for_child = current_node.data[positions]
                else:
                    positions = np.where(length(current_node.data[:, 0]) > long_column_limit)
                    data_for_child = current_node.data[positions]
                    breaking_flag = True
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
            if breaking_flag:
                break
        current_node.children = child_list
        if len(root.leaves) > limit:
            break
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
    gc.collect()
    scoring_matrix = np.empty([len(leaves), len(leaves)], dtype='float64')

    # TODO: Maybe parallelise it here?
    for i in range(len(leaves)):
        for j in range(i, len(leaves)):
            masses_difference = abs(leaves[i].data.shape[0] - leaves[j].data.shape[0]) + 1
            if i == j:
                scoring_matrix[i][j] = 0
            else:
                # masses multiplication
                masses_multiplication = (leaves[i].data.shape[0] * leaves[j].data.shape[0])

                # calculate distance
                distance = (node_distance(leaves[i], leaves[j], max_depth)) ** 2
                scoring_matrix[i][j] = masses_multiplication * (1 / distance) * (1 / masses_difference)

    matrix = scoring_matrix.T + scoring_matrix
    return matrix


def fibonacci_generator():
    a, b = 2, 3
    while True:
        yield a
        a, b = b, a + b


def process_attribute(attribute_to_process: str, dataframe: pd.DataFrame):
    column = attribute_to_process
    attribute = process_data(pd.DataFrame(dataframe[column]))
    root = tree_grow(attribute)
    ndistinct = 2
    fibonacci = fibonacci_generator()
    tries = 0  # failsafe mechanism
    while True or tries < 40:
        leaves = root.leaves
        logger.debug(f"LENGTH OF LEAVES IS {len(leaves)}")
        if len(leaves) < calculate_machine_limit():
            break
        else:
            ndistinct = next(fibonacci)
            logger.debug("Didn't manage to fit a tree. Building another one")
            root = tree_grow(attribute, nDistinctMin=ndistinct)
            tries += 1
    logger.debug("N distinct value is " + str(ndistinct))
    logger.debug(attribute)
    score_matrix = score_function(leaves)
    medians = np.ma.median(np.ma.masked_invalid(score_matrix, 0), axis=1).data
    output_dictionary = {}
    outlying_elements = {}
    pattern_elements = {}

    threshold_values = np.linspace(1, 99, 5000)
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


def over_the_limit_check():
    return True


def replace_repeated_chars(string):
    if len(string) < 2:
        return string

    result = [string[0]]  # Add the first character to the result list
    asterisk_added = False

    for i in range(1, len(string)):
        if string[i] != string[i - 1]:
            result.append(string[i])
            asterisk_added = False
        elif not asterisk_added:
            result.append('*')
            asterisk_added = True

    return ''.join(result)


def choose_string_generalise_method():
    return True


def add_outlying_elements_to_attribute(column: str, dataframe: pd.DataFrame):
    col_outliers_and_patterns = process_attribute(column, dataframe)
    lexicon = {column: {'outliers': {}, 'patterns': {}}}
    has_previous_threshold_dict = False
    previous_threshold_dict_value = -1
    marked_for_clearance = []
    generalised_strings_ratio_sum = 0
    generalised_strings_ratio_counter = 0
    regexed_strings_ratio_sum = 0
    regexed_strings_ratio_counter = 0
    for threshold_level in col_outliers_and_patterns['outliers'].keys():
        if threshold_level < 50:
            lexicon[column]['outliers'][threshold_level] = {}
            inner_dicts = col_outliers_and_patterns['patterns'][threshold_level]
            pattern_set_generalised = set()
            pattern_set_regexed = set()
            for representation in inner_dicts.keys():
                pattern_set_regexed.add(
                    replace_repeated_chars(generalise_string(next(iter(inner_dicts[representation])))))
                pattern_set_generalised.add(generalise_string(next(iter(inner_dicts[representation]))))
            generalised_string_in_pattern_set = 0
            regexed_string_in_pattern_set = 0
            generalised_string_not_in_pattern_set = 0
            regexed_string_not_in_pattern_set = 0
            for outlier_rep in col_outliers_and_patterns['outliers'][threshold_level].keys():
                first_outliers_element = list(col_outliers_and_patterns['outliers'][threshold_level][outlier_rep])[0]
                generalised_pattern = generalise_string(first_outliers_element)
                regexed_string = replace_repeated_chars(generalised_pattern)
                if regexed_string in pattern_set_regexed:
                    regexed_string_in_pattern_set += 1
                    continue
                else:
                    regexed_string_not_in_pattern_set += 1

                if generalised_pattern in pattern_set_generalised:
                    generalised_string_in_pattern_set += 1
                    continue
                else:
                    generalised_string_not_in_pattern_set += 1
            logger.debug("Threshold level:" + str(threshold_level))
            try:

                ratio_generalised = generalised_string_in_pattern_set / (
                        generalised_string_in_pattern_set + generalised_string_not_in_pattern_set)
                generalised_strings_ratio_sum += ratio_generalised
                generalised_strings_ratio_counter += 1
                logger.debug("Generalised Ratio: " + str(ratio_generalised))

            except:
                logger.debug(" ")

            try:
                ratio_regexed = regexed_string_in_pattern_set / (
                        regexed_string_in_pattern_set + regexed_string_not_in_pattern_set)
                regexed_strings_ratio_sum += ratio_regexed
                regexed_strings_ratio_counter += 1
                logger.debug("Regexed ratio: " + str(ratio_regexed))

            except:
                logger.debug(" ")
    try:
        avg_regex_ratio = regexed_strings_ratio_sum / regexed_strings_ratio_counter
    except:
        avg_regex_ratio = 0
    try:
        avg_generalised_ratio = generalised_strings_ratio_sum / generalised_strings_ratio_counter
    except:
        avg_generalised_ratio = 0
    logger.debug("AVG REGEX RATIO IS " + str(avg_regex_ratio))
    logger.debug("AVG GENERALISED RATIO IS " + str(avg_generalised_ratio))
    # if both zero then choose generalised.
    apply_generalised_comparison = True
    if avg_regex_ratio > avg_generalised_ratio and avg_regex_ratio < 0.98 and avg_regex_ratio > 0.42:
        apply_generalised_comparison = False
    logger.debug("Applying Generalised Comparison: " + str(apply_generalised_comparison))

    for threshold_level in col_outliers_and_patterns['outliers'].keys():
        if threshold_level < list(col_outliers_and_patterns['outliers'].keys())[len(col_outliers_and_patterns['outliers'].keys()) // 2]:
            lexicon[column]['outliers'][threshold_level] = {}
            inner_dicts = col_outliers_and_patterns['patterns'][threshold_level]
            pattern_set = set()
            for representation in inner_dicts.keys():
                if apply_generalised_comparison:
                    pattern_set.add(generalise_string(next(iter(inner_dicts[representation]))))

                else:
                    pattern_set.add(replace_repeated_chars(generalise_string(next(iter(inner_dicts[representation])))))
            for outlier_rep in col_outliers_and_patterns['outliers'][threshold_level].keys():
                first_outliers_element = list(col_outliers_and_patterns['outliers'][threshold_level][outlier_rep])[0]
                transformed_string = generalise_string(first_outliers_element)
                if not apply_generalised_comparison:
                    transformed_string = replace_repeated_chars(transformed_string)

                if transformed_string in pattern_set:
                    continue
                else:
                    inner_dict = lexicon[column]['outliers'][threshold_level].get(transformed_string, {})
                    inner_dict.update(col_outliers_and_patterns['outliers'][threshold_level][outlier_rep])
                    lexicon[column]['outliers'][threshold_level][transformed_string] = inner_dict

            if has_previous_threshold_dict:
                current_dict = lexicon[column]['outliers'][threshold_level]
                previous_dict = lexicon[column]['outliers'][previous_threshold_dict_value]
                if compare_dicts(current_dict, previous_dict):
                    marked_for_clearance.append(threshold_level)
            has_previous_threshold_dict = True
            previous_threshold_dict_value = threshold_level
    for threshold_level in marked_for_clearance:
        del lexicon[column]['outliers'][threshold_level]

    # Now for the patterns
    has_previous_threshold_dict = False
    previous_threshold_dict_value = -1
    marked_for_clearance = []
    for threshold_level in sorted(col_outliers_and_patterns['patterns'].keys(), reverse=True):
        if threshold_level >= list(col_outliers_and_patterns['outliers'].keys())[len(col_outliers_and_patterns['outliers'].keys()) // 2]:
            lexicon[column]['patterns'][threshold_level] = {}
            inner_dicts = col_outliers_and_patterns['patterns'][threshold_level]
            pattern_set = set()
            for representation in inner_dicts.keys():
                if apply_generalised_comparison:
                    pattern_set.add(generalise_string(next(iter(inner_dicts[representation]))))
                else:
                    pattern_set.add(replace_repeated_chars(generalise_string(next(iter(inner_dicts[representation])))))
            for pattern in pattern_set:
                lexicon[column]['patterns'][threshold_level][pattern] = {}
            for pattern_rep in col_outliers_and_patterns['patterns'][threshold_level].keys():
                first_patterns_element = list(col_outliers_and_patterns['patterns'][threshold_level][pattern_rep])[0]
                transformed_string = generalise_string(first_patterns_element)
                if not apply_generalised_comparison:
                    transformed_string = replace_repeated_chars(transformed_string)

                inner_dict = lexicon[column]['patterns'][threshold_level].get(transformed_string, {})
                inner_dict.update(col_outliers_and_patterns['patterns'][threshold_level][pattern_rep])
                lexicon[column]['patterns'][threshold_level][transformed_string] = inner_dict

            if has_previous_threshold_dict:
                current_dict = lexicon[column]['patterns'][threshold_level]
                previous_dict = lexicon[column]['patterns'][previous_threshold_dict_value]
                if compare_dicts(current_dict, previous_dict):
                    marked_for_clearance.append(threshold_level)
            has_previous_threshold_dict = True
            previous_threshold_dict_value = threshold_level
    for threshold_level in marked_for_clearance:
        del lexicon[column]['patterns'][threshold_level]

    logger.debug("Finished " + column)
    return lexicon


def process_column(column, dataframe):
    try:
        return add_outlying_elements_to_attribute(column, dataframe), {}
    except MemoryError:
        logger.debug("Out of memory error occurred for column:", column)
        return {}, column


def process(file: str, multiprocess_switch):
    try:
        dataframe = read_data("resources/json_dumps/" + file.filename)
    except AttributeError:
        # dataframe = read_data("../resources/datasets/datasets_testing_purposes/testing123.csv")
        # dataframe = read_data("resources/datasets/datasets_testing_purposes/dirty.csv")
        # dataframe = read_data("resources/json_dumps/flightsDirty.csv")
        # dataframe = read_data("../resources/json_dumps/School_Learning_Modalities__2020-2021.csv")
        # dataframe = read_data("../resources/json_dumps/pima-indians-diabetes.csv")
        # dataframe = read_data("../resources/datasets/datasets_testing_purposes/10492-1.csv")
        # dataframe = read_data("../resources/datasets/datasets_testing_purposes/16834-1.csv")
        # dataframe = read_data("../resources/datasets/datasets_testing_purposes/adult.csv")
        # dataframe = read_data("resources/datasets/datasets_testing_purposes/hospital/HospitalClean.csv")
        # dataframe = read_data("resources/datasets/datasets_testing_purposes/tax/taxClean.csv")
        # dataframe = read_data("resources/datasets/datasets_testing_purposes/flights/flightsDirty.csv")
        # dataframe = read_data("resources/datasets/datasets_testing_purposes/beers/beersClean.csv")
        # dataframe = read_data("resources/datasets/datasets_testing_purposes/toy/toyDirty.csv")
        # dataframe = read_data("resources/datasets/datasets_testing_purposes/Lottery_Powerball_Winning_Numbers__Beginning_2010.csv")
        dataframe = read_data("resources/datasets/datasets_testing_purposes/movies_1/moviesDirty.csv")
        # dataframe = read_data("resources/datasets/datasets_testing_purposes/banklist.csv")
        # dataframe = read_data("resources/datasets/datasets_testing_purposes/Air_Traffic_Passenger_Statistics.csv")
        # dataframe = read_data("resources/datasets/datasets_testing_purposes/testing123.csv")

    with joblib.parallel_backend("loky"):
        results = Parallel(n_jobs=-1)(
            delayed(process_column)(column, dataframe) for column in dataframe.columns
        )
    output = {}
    error_columns = []

    for result in results:
        column_result, error_column = result
        if error_column:
            error_columns.append(error_column)
        else:
            output.update(column_result)
    for column in error_columns:
        logger.debug("Computing the columns that errored")
        output.update(add_outlying_elements_to_attribute(column, dataframe))

    return output
    # output = {}
    # for column in dataframe.columns:
    #     if column != "acssstors":
    #         output.update(add_outlying_elements_to_attribute(column, dataframe))
    # return output
