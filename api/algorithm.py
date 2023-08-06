import gc
import logging
from datetime import datetime

import chardet
import joblib
import numpy as np
import pandas as pd
from anytree import Node
from doubledouble import DoubleDouble
from joblib import delayed, Parallel

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
memory_problems = False
ndistinct_manually_set = False
ndistinct_manual_setting = 2
large_file = False
long_column_limit = 36  # This is based on the length of a UUID
large_file_threshold = 500000


def read_data(filename, column_name=None):
    file_extension = filename.split(".")[-1]
    if file_extension == "csv":
        with open(filename, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding']
        if column_name is not None:
            return pd.read_csv(filename, dtype=str, encoding=encoding, usecols=[column_name])
        else:
            return pd.read_csv(filename, dtype=str, encoding=encoding)
    elif file_extension == "xlsx":
        return pd.read_excel(filename, dtype=str)
    elif file_extension == "json":
        return pd.read_json(filename)
    elif file_extension == "tsv":
        return pd.read_csv(filename, sep='\t', dtype=str)
    else:
        # TODO: Add database support here in the future
        raise ValueError("Unsupported file format. Only CSV, XLSX, and JSON formats are supported.")


def process_data(dataframe: pd.DataFrame):
    if type(dataframe) is pd.Series:
        dataframe = dataframe.to_frame()
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
    This function determines the kind of split that will be applied on the dataset at hand. The split dependent
    on the specificity_level (i.e. the depth of the tree).
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

        return set(word_split(df[:, 0]))


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
        # take a lower bound to make room for the other variables as well.
        if memory_problems:
            return i - 8000
        else:
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
    root = Node(" ", children=[], data=np.asarray(column), specificity_level=-2)
    node_list = [root]
    global long_column_limit
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
            return root
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


def score_sensitive_function(leaves):
    """
    There are cases where a small dataset produces numerical instability problems and forces an overflow or an underflow.
    DoubleDouble is thus used to solve this issue.
    :param leaves:
    :return:
    """
    max_depth = max([leaf.depth for leaf in leaves])

    scoring_matrix = np.empty([len(leaves), len(leaves)], dtype=object)
    scoring_matrix.fill(0)

    for i in range(len(leaves)):
        for j in range(i, len(leaves)):
            masses_difference = abs(leaves[i].data.shape[0] - leaves[j].data.shape[0]) + 1
            if i == j:
                scoring_matrix[i][j] = DoubleDouble(0)
            else:
                masses_multiplication = DoubleDouble(leaves[i].data.shape[0] * leaves[j].data.shape[0])

                distance = DoubleDouble(node_distance(leaves[i], leaves[j], max_depth)) ** 2
                scoring_matrix[i][j] = masses_multiplication * (DoubleDouble(1) / distance) * (
                        DoubleDouble(1) / masses_difference)

    scoring_matrix = np.where(scoring_matrix == None, 0, scoring_matrix)
    matrix = scoring_matrix.T + scoring_matrix
    return matrix


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
    score_flag = False

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
                score = masses_multiplication * (1 / distance) * (1 / masses_difference)
                if score < 0 or score > 100000000000000000:
                    score_flag = True
                scoring_matrix[i][j] = score
    if score_flag:
        return score_sensitive_function(leaves)
    matrix = scoring_matrix.T + scoring_matrix
    return matrix


def fibonacci_generator():
    """
    A fibonacci generator in order to automatically increase the hyperparameter ndistinctMin,
    :return: It yields the next fibonacci. It does not return it.
    """
    a, b = 2, 3
    while True:
        yield a
        a, b = b, a + b


def median_vector_can_fit(score_matrix):
    """
    Memory check helper function. Checks if the median vector can fit in memory by forcing a memory overflow.
    :param score_matrix: The score matrix created by the algorithm.
    :return: Boolean flag that indicates whether the median can fit in main memory.
    """
    try:
        medians = np.ma.median(np.ma.masked_invalid(score_matrix, 0), axis=1).data
        gc.collect()
        return True
    except:
        gc.collect()
        return False


def process_attribute(dataframe: pd.DataFrame):
    global ndistinct_manually_set
    attribute = process_data(dataframe)
    if not ndistinct_manually_set:
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
                # if the file is large then going to the next fibonacci won't do much.
                # But going five steps further has bigger effect
                if large_file:
                    ndistinct = next(fibonacci)
                    ndistinct = next(fibonacci)
                    ndistinct = next(fibonacci)
                    ndistinct = next(fibonacci)
                ndistinct = next(fibonacci)
                logger.debug(f"Didn't manage to fit a tree with ndistinct={ndistinct}. Building another one")
                root = tree_grow(attribute, nDistinctMin=ndistinct)
                tries += 1

    else:
        logger.debug("Reminder that ndistinct is " + str(ndistinct_manual_setting))
        root = tree_grow(attribute, nDistinctMin=ndistinct_manual_setting)
        leaves = root.leaves
        logger.debug(f"Length of leaves is {len(leaves)}")

    score_matrix = score_function(leaves)
    try:
        medians = np.ma.median(np.ma.masked_invalid(score_matrix, 0), axis=1).data
    except:
        logger.debug("Median vector cannot fit. Increasing ndistinct...")
        ndistinct = next(fibonacci)
        root = tree_grow(attribute, nDistinctMin=ndistinct)
        leaves = root.leaves
        score_matrix = score_function(leaves)
        while not median_vector_can_fit(score_matrix):
            if tries > 80:
                break
            ndistinct = next(fibonacci)  # Skip a level and directly go to the next three steps
            ndistinct = next(fibonacci)  # We are probably talking about millions of lines
            ndistinct = next(fibonacci)
            logger.debug("Didn't manage to fit the median. Building another tree")
            root = tree_grow(attribute, nDistinctMin=ndistinct)
            leaves = root.leaves
            score_matrix = score_function(leaves)
            tries += 1
        medians = np.ma.median(np.ma.masked_invalid(score_matrix, 0), axis=1).data

    logger.debug(attribute)
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
        lower_outlying_indices = np.argwhere(medians <= outlier_threshold)
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
    """
    Helper function. Check if two dicts are the same or not.
    :param dict1: A dictionary
    :param dict2: Another dictionary
    :return: Boolean expression
    """
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


def replace_repeated_chars(string):
    """
    Create the regexed expression for a given generalised string, where the consecutively repeated characters
     are turned into +
    For example, string UUUUddsU turns into U+d+sU.
    :param string: A generalised string
    :return: A regexed string
    """
    if len(string) < 2:
        return string

    result = [string[0]]  # Add the first character to the result list
    asterisk_added = False

    for i in range(1, len(string)):
        if string[i] != string[i - 1]:
            result.append(string[i])
            asterisk_added = False
        elif not asterisk_added:
            result.append('+')
            asterisk_added = True

    return ''.join(result)


def find_difference_index(str1, str2):
    """
    Helper function that finds which index two strings differ
    :param str1: A string
    :param str2: Another string
    :return: The index that two strings differ. Return -1 if they don't
    """
    str1 = str(str1)
    str2 = str(str2)

    length = min(len(str1), len(str2))

    for i in range(length):
        if str1[i] != str2[i]:
            return i

    if len(str1) != len(str2):
        return length

    return -1


def merge_dictionaries(dict1, dict2):
    """
    Takes two dictionaries and merges them. Helper function
    :param dict1: a dictionary
    :param dict2: another dictionary
    :return: Merged dict that contains both dict1 and dict2
    """
    merged_dict = {}

    # Merge dictionaries from dict1
    for key, value in dict1.items():
        if key in dict2:
            merged_subdict = {}
            for subkey, subvalue in value.items():
                if subkey in dict2[key]:
                    merged_subdict[subkey] = subvalue + dict2[key][subkey]
                else:
                    merged_subdict[subkey] = subvalue
            for subkey, subvalue in dict2[key].items():
                if subkey not in merged_subdict:
                    merged_subdict[subkey] = subvalue
            merged_dict[key] = merged_subdict
        else:
            merged_dict[key] = value

    # Add dictionaries from dict2 that don't exist in dict1
    for key, value in dict2.items():
        if key not in merged_dict:
            merged_dict[key] = value

    return merged_dict


def convert_to_percentage(data, number):
    """
    Recursive function which takes the deep dictionary that the algorithm has detected and turns the occurrences (which
    are the innermost values) to percentages, so it can be used in the front end.
    :param data: the dictionary
    :param number: the length of the attribute
    :return: the same dictionary but instead of occurrences for each key, they are percentages relative to the
            attribute's length
    """
    if isinstance(data, dict):
        new_dict = {}
        for key, value in data.items():
            new_dict[key] = convert_to_percentage(value, number)
        return new_dict
    elif isinstance(data, int):
        return data / number
    else:
        return data


def add_outlying_elements_to_attribute(column_name: str, dataframe_column: pd.DataFrame):
    col_outliers_and_patterns = process_attribute(dataframe_column)
    lexicon = {column_name: {'outliers': {}, 'patterns': {}}}
    has_previous_threshold_dict = False
    previous_threshold_dict_value = -1
    marked_for_clearance = []
    generalised_strings_ratio_sum = 0
    generalised_strings_ratio_counter = 0
    regexed_strings_ratio_sum = 0
    regexed_strings_ratio_counter = 0
    for threshold_level in col_outliers_and_patterns['outliers'].keys():
        if threshold_level < list(col_outliers_and_patterns['outliers'].keys())[
            len(col_outliers_and_patterns['outliers'].keys()) // 2] and len(
            list(col_outliers_and_patterns['outliers'].keys())) > 1:
            lexicon[column_name]['outliers'][threshold_level] = {}
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
            try:

                ratio_generalised = generalised_string_in_pattern_set / (
                        generalised_string_in_pattern_set + generalised_string_not_in_pattern_set)
                generalised_strings_ratio_sum += ratio_generalised
                generalised_strings_ratio_counter += 1

            except:
                logger.debug(" ")

            try:
                ratio_regexed = regexed_string_in_pattern_set / (
                        regexed_string_in_pattern_set + regexed_string_not_in_pattern_set)
                regexed_strings_ratio_sum += ratio_regexed
                regexed_strings_ratio_counter += 1

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
        if threshold_level < list(col_outliers_and_patterns['outliers'].keys())[
            len(col_outliers_and_patterns['outliers'].keys()) // 2] and len(
            list(col_outliers_and_patterns['outliers'].keys())) > 1:
            lexicon[column_name]['outliers'][threshold_level] = {}
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
                    inner_dict = lexicon[column_name]['outliers'][threshold_level].get(transformed_string, {})
                    inner_dict.update(col_outliers_and_patterns['outliers'][threshold_level][outlier_rep])
                    lexicon[column_name]['outliers'][threshold_level][transformed_string] = inner_dict

            if has_previous_threshold_dict:
                current_dict = lexicon[column_name]['outliers'][threshold_level]
                previous_dict = lexicon[column_name]['outliers'][previous_threshold_dict_value]
                if compare_dicts(current_dict, previous_dict):
                    marked_for_clearance.append(threshold_level)
            has_previous_threshold_dict = True
            previous_threshold_dict_value = threshold_level
    for threshold_level in marked_for_clearance:
        del lexicon[column_name]['outliers'][threshold_level]

    # Now for the patterns
    has_previous_threshold_dict = False
    previous_threshold_dict_value = -1
    marked_for_clearance = []
    helper_dict = {}
    for threshold_level in sorted(col_outliers_and_patterns['patterns'].keys(), reverse=True):
        if threshold_level >= list(col_outliers_and_patterns['outliers'].keys())[
            len(col_outliers_and_patterns['outliers'].keys()) // 2]:
            helper_dict[threshold_level] = {}
            lexicon[column_name]['patterns'][threshold_level] = {}
            pattern_set = set()
            for pattern_rep in col_outliers_and_patterns['patterns'][threshold_level].keys():
                first_patterns_element = list(col_outliers_and_patterns['patterns'][threshold_level][pattern_rep])[0]

                transformed_string = generalise_string(first_patterns_element)

                if not apply_generalised_comparison:
                    transformed_string = replace_repeated_chars(transformed_string)

                if not bool(lexicon[column_name]['patterns'][threshold_level].get(transformed_string, {})):
                    lexicon[column_name]['patterns'][threshold_level][transformed_string] = {}
                if not bool(lexicon[column_name]['patterns'][threshold_level][transformed_string].get(pattern_rep, {})):
                    lexicon[column_name]['patterns'][threshold_level][transformed_string][pattern_rep] = {}
                temp_pat_dict = col_outliers_and_patterns['patterns'][threshold_level][pattern_rep]
                lexicon[column_name]['patterns'][threshold_level][transformed_string][pattern_rep].update(temp_pat_dict)
                pattern_set.add(transformed_string)

            for outlier_representative in col_outliers_and_patterns['outliers'][threshold_level]:
                first_outliers_element = \
                    list(col_outliers_and_patterns['outliers'][threshold_level][outlier_representative])[0]

                transformed_string = generalise_string(first_outliers_element)
                if not apply_generalised_comparison:
                    transformed_string = replace_repeated_chars(transformed_string)
                if transformed_string in pattern_set:
                    if not bool(helper_dict[threshold_level].get(transformed_string, {})):
                        helper_dict[threshold_level][transformed_string] = {}
                    if not bool(helper_dict[threshold_level][transformed_string].get(outlier_representative, {})):
                        helper_dict[threshold_level][transformed_string][outlier_representative] = {}
                    temp_dict = col_outliers_and_patterns['outliers'][threshold_level][outlier_representative]
                    helper_dict[threshold_level][transformed_string][outlier_representative].update(temp_dict)

            if has_previous_threshold_dict:
                current_dict = lexicon[column_name]['patterns'][threshold_level]
                previous_dict = lexicon[column_name]['patterns'][previous_threshold_dict_value]
                if compare_dicts(current_dict, previous_dict):
                    marked_for_clearance.append(threshold_level)
            has_previous_threshold_dict = True
            previous_threshold_dict_value = threshold_level
    for threshold_level in marked_for_clearance:
        del lexicon[column_name]['patterns'][threshold_level]
    total_number_of_elements_in_database = dataframe_column.shape[0]
    for threshold_level in lexicon[column_name]['patterns']:
        lexicon[column_name]['patterns'][threshold_level] = merge_dictionaries(helper_dict[threshold_level],
                                                                               lexicon[column_name]['patterns'][
                                                                                   threshold_level])
    lexicon[column_name]['patterns'] = convert_to_percentage(lexicon[column_name]['patterns'],
                                                             total_number_of_elements_in_database)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.debug(f'The current time is: {current_time}')
    logger.debug("Finished " + column_name)
    return lexicon


def process_column(column_name, single_column):
    try:
        return add_outlying_elements_to_attribute(column_name, single_column), {}
    except MemoryError:
        logger.debug("Out of memory error occurred for column:" + str(column_name))
        return {}, column_name


def reset_global_values():
    global ndistinct_manually_set
    global memory_problems
    global ndistinct_manual_setting
    global large_file
    global long_column_limit
    global large_file_threshold

    memory_problems = False
    ndistinct_manually_set = False
    ndistinct_manual_setting = 0
    large_file = False
    long_column_limit = 36
    large_file_threshold = 500000


def set_global_variables(manual_override_ndistinct):
    global ndistinct_manually_set
    global ndistinct_manual_setting
    ndistinct_manually_set = True
    ndistinct_manual_setting = manual_override_ndistinct
    return None


def process(file, manual_override_ndistinct=None, first_time=True, column_name=None, manual_override_long_column=None,
            manual_override_large_file_threshold=None):
    global large_file_threshold
    if manual_override_large_file_threshold is not None:
        large_file_threshold = manual_override_large_file_threshold
    global long_column_limit
    if manual_override_long_column is not None:
        long_column_limit = manual_override_long_column
    try:
        dataframe = read_data("resources/data_repository/" + file.filename, column_name)
    except AttributeError:
        dataframe = read_data("resources/datasets/datasets_testing_purposes/toy/toyDirty.csv")

    if dataframe.shape[0] > large_file_threshold and first_time:  # if dataframe too large
        logger.debug("Large file detected")
        return list(dataframe.columns)
    if dataframe.shape[0] < large_file_threshold:
        with joblib.parallel_backend("loky"):
            try:
                if manual_override_ndistinct is not None:
                    set_global_variables(manual_override_ndistinct)
                    output = {}
                    for column in dataframe.columns:
                        single_column = dataframe[column]
                        output.update(process_column(column, single_column)[0])
                    return output
                else:
                    results = Parallel(n_jobs=-1)(
                        delayed(process_column)(column, dataframe[column]) for column in dataframe.columns
                    )
            except:
                output = {}
                for column in dataframe.columns:
                    single_column = dataframe[column]
                    output.update(process_column(column, single_column)[0])
                return output
    else:
        global large_file
        large_file = True
        if manual_override_ndistinct is not None:
            set_global_variables(manual_override_ndistinct)
            logger.debug(f"Reminder that ndistinctMin is {manual_override_ndistinct}")
        output = {}
        for column in dataframe.columns:
            single_column = dataframe[column]
            output.update(process_column(column, single_column)[0])
        reset_global_values()
        return output
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
        single_column = dataframe[column]
        try:
            output.update(add_outlying_elements_to_attribute(column, single_column))
        except:
            global memory_problems
            memory_problems = True
            output.update(add_outlying_elements_to_attribute(column, single_column))
    reset_global_values()
    return output
