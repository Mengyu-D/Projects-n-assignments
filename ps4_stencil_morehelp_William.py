# If you consider packages or modules other than the following imported
# as absolutely necessary, then please contact the teaching team.
# Note also that `pandas` is allowed, but only when using it together with
# `matplotlib` or `plotnine`.

import csv
import math
import numpy as np
import collections
import pprint
import pandas as pd
import matplotlib
import plotnine as p9

"""
INAFU6503: Applying Machine Learning
Problem Set 4: Prediction
Professor Daniel Bjorkegren
Columbia University


Fill in the functions `load_data`, `partition_loss`, `build_tree`, and
`predict` and complete parts where it says "TODO" as needed to complete the
tasks in the problem set documnetation.

This is a version of the code that has much more stencil
code for you to work with. If you feel that this is not too challenging,
please take a look at "ps4_stencil.py" and work with the said file.
"""


# ------------------- READING IN DATA ------------------------------------

"""************************************************************************
* function:  load_data(for_prediction)
* arguments:
*       -for_prediction: boolean that is "True" if reading in unlabeled data for
*                        making predictions (i.e., 'loans_B_unlabeled.csv'),
*                        "False" if reading in labeled data to build a 
*                        regression tree (i.e., 'loans_A_labeled.csv')
* return value:  a list of tuples representing the labeled loans data if
*                `for_prediction` is False; a list of dictionaries to represent
*                features of unlabeled loans data if `for_prediction` is True
*
* TODO:  Read in the loans data from the provided csv file. If `for_prediction`
*        is False, store the observations as a list of tuples (a, b), where 'a'
*        is a dictionary of features and 'b' is the value of the
*        `days_until_funded` variable. If `for_prediction` is True, store the
*        observations as a list of dictionaries where each dictionary is that
*        of features.
************************************************************************"""

        # TODO: use `dict_reader` to read in each observation as formatted
        # in the function documentation, and append to the list `data`.
        # Also, if necessary, make adjustments here to create additional
        # features based on pre-existing features that will be used in your
        # regression tree.

def load_data(for_prediction=False):
    file = "/Users/williamsempire/Downloads/PS4/tables/loans_A_labeled.csv"
    if for_prediction:
        file = "/Users/williamsempire/Downloads/PS4/tables/loans_B_unlabeled.csv"
    data = []

    with open(file, "rt", encoding="utf8") as f:
        dict_reader = csv.DictReader(f)
        for observation in dict_reader:
            feature_dict = {}

            for key, value in observation.items():
                feature_dict[key] = value

            if for_prediction == False:
                data.append((feature_dict, observation["days_until_funded"]))

            elif for_prediction == True:
                data.append(feature_dict)
    return data

LoanA_labeled = load_data(for_prediction=False)


# extracts a list of features to compare to days_until_funded
# returns a list of tuples containing a dictionary of features and days_until_funded
# maybe use this to analyze relationships between these features and days_until_funded
def get_feature(for_prediction=False):
    file = "/Users/williamsempire/Downloads/PS4/tables/loans_A_labeled.csv"
    if for_prediction:
        file = "/Users/williamsempire/Downloads/PS4/tables/loans_B_unlabeled.csv"
    data = []

    with open(file, "rt", encoding="utf8") as f:
        dict_reader = csv.DictReader(f)

        for observation in dict_reader:
            feature_dict = {"loan_amount": observation["loan_amount"],
                            "activity": observation["activity"],
                            "sector": observation["sector"]
                            }

            if not for_prediction:
                data.append((feature_dict, {"days_until_funded": observation["days_until_funded"]}))
            else:
                data.append(feature_dict)

    return data

# result2 = get_feature(for_prediction=False)
# print(result2)


def group_feature(df, n_bins):
    # 1. Group by sector
    grouped_sector = (df.groupby("sector")["days_until_funded"]
          .mean()
          .reset_index()
          .sort_values("days_until_funded"))
    
    # 2. Group by activity
    grouped_activity = (df.groupby("activity")["days_until_funded"]
          .mean()
          .reset_index()
          .sort_values("days_until_funded"))
    
    # 3. Bin loan_amount into quantile-based bins (e.g., quartiles)
    df['loan_amount_bin'] = pd.qcut(df['loan_amount'], n_bins,
        labels=[f'Q{i+1}' for i in range(n_bins)])
    
    grouped_loan_amount = (df.groupby("loan_amount_bin")["days_until_funded"]
          .mean()
          .reset_index()
          .sort_values("days_until_funded"))
    
    return ({"sector": grouped_sector}, 
            {"activity": grouped_activity},
            {"loan_amount": grouped_loan_amount})

if __name__ == "__main__":
    file = "/Users/williamsempire/Downloads/PS4/tables/loans_A_labeled.csv"
    df = pd.read_csv(file)

    feature_avg = group_feature(df, n_bins=4)
    print(feature_avg)



# ------------------- CREATING ADDITIONAL FEATURES ---------------------------
"""
* TODO: Create features to be used in your regression tree.
* 
* As an example for dealing with continuous variables, function
* `continuous_to_percentile` is provided below. This function is by no means
* an exhaustive method to deal with continuous or other types of variables in
* the dataset.
"""

"""************************************************************************
* function:  continuous_to_percentile(observations, continuous_var, n_bins)
* arguments:
*       -observations: a list of tuples (a, b) representing loans data, where
*                      'a' is a dictionary of features and 'b' is the value of
*                      the `days_until_funded` variable
*       -continuous_var: string representing the feature whose values are
*                        considered continuous rather than binary or
*                        categorical
*       -n_bins: integer to indicate how many "bins" of percentiles should be
*                created for the continuous variable. For instance, if it is
*                4, the function will create quartiles (i.e., 0-25th, 25-50th,
*                50-75th, and 75-100th percentiles) and create a binary
*                variable each for a quartile that equals to 1 if it falls
*                within the said quartile.
* 
* example use case:
*     # assuming `load_data` function was written correctly
*     data = load_data()
*     
*     # example with the variable "loan_amount"
*     modified_data = continuous_to_percentile(data, "loan_amount", 4)
*
* return value:  a list of tuples representing the loans data, but with the
*                specified 'continuous_var' replaced by binary variables
*                as written in the description for 'n_bins'.
************************************************************************"""


def continuous_to_percentile(observations, continuous_var, n_bins=4):
    var_values = [float(obs[0][continuous_var]) for obs in observations]
    percentiles_to_calc = [(i + 1) * 100 / n_bins for i in range(0, n_bins)]
    percentiles = np.percentile(var_values, percentiles_to_calc)
    new_var_names = [f"{continuous_var}_{i}_{n_bins}" for i in range(1, n_bins + 1)]

    new_data = []
    for obs in observations:
        features, days_until_funded = obs

        # copying, otherwise the original dataset will be modified
        features_to_modify = features.copy()
        var_value = float(features_to_modify.pop(continuous_var))

        for var in new_var_names:
            features_to_modify[var] = 0

        for i, percentile in enumerate(percentiles):
            var_name = new_var_names[i]
            if i == 0:
                if var_value <= percentile:
                    features_to_modify[var_name] = 1
            else:
                lower_bound = percentiles[i - 1]
                if (var_value > lower_bound) and (var_value <= percentile):
                    features_to_modify[var_name] = 1

        new_data.append((features_to_modify, days_until_funded))

    return new_data



# modified_data =  continuous_to_percentile(data, "loan_amount", n_bins=4)
# print(modified_data)

# -------------------- BUILDING BLOCKS FOR TREES -------------------------

"""************************************************************************
* function: partition_loss(subsets)
* arguments:
* 		-subsets:  a list of lists of labeled data (representing groups
				   of observations formed by a split)
* return value:  loss value of a partition into the given subsets
*
* TODO: Write a function that computes the loss of a partition for
*       given subsets
************************************************************************"""


def partition_loss(subsets):
    # TODO
    pass


"""************************************************************************
* function: partition_by(inputs, attribute)
* arguments:
* 		-inputs:  a list of observations in the form of tuples
*		-attribute:  an attribute on which to split
* return value:  a list of lists, where each list represents a subset of
*				 the inputs that share a common value of the given 
*				 attribute
************************************************************************"""


def partition_by(inputs, attribute):
    groups = collections.defaultdict(list)
    for input_ in inputs:
        key = input_[0][attribute]  # gets the value of the specified attribute
        groups[key].append(input_)  # add the input to the appropriate group
    return groups


"""************************************************************************
* function: partition_loss_by(inputs, attribute)
* arguments:
* 		-inputs:  a list of observations in the form of tuples
*		-attribute:  an attribute on which to split
* return value:  the loss value of splitting the inputs based on the
*				 given attribute
************************************************************************"""


def partition_loss_by(inputs, attribute):
    partitions = partition_by(inputs, attribute)
    return partition_loss(partitions.values())


# -------------------- BUILDING A REGRESSION TREE -------------------------


"""************************************************************************
* function:  build_tree(inputs, num_levels, split_candidates=None)
*
* arguments:
* 		-inputs:  labeled data used to construct the tree; should be in the
*				  form of a list of tuples (a, b) where 'a' is a dictionary
*				  of features and 'b' is a label
*		-num_levels:  the goal number of levels for our output tree
*		-split_candidates:  variables that we could possibly split on.  For
*							our first level, all variables are candidates
*							(see first two lines in the function).
*			
* return value:  a tree in the form of a tuple (a, b) where 'a' is the
*				 variable to split on and 'b' is a dictionary representing
*				 the outcome class/outcome for each value of 'a'.
* 
* TODO:  Write a recursive function that builds a REGRESSION tree (NOT a
*        classification tree!) of the specified number of levels based on
*        labeled data "inputs"
************************************************************************"""


def build_tree(inputs, num_levels, split_candidates=None):
    # if first pass, all keys are split candidates
    if split_candidates == None:
        split_candidates = inputs[0][0].keys()
    split_candidates = list(split_candidates)

    # TODO: process the cases where the current subset should considered as
    # a leaf node. In these cases, you should simply return the predicted value
    # from the subset.

    # TODO: use a for-loop over each attribute, and append each resulting loss
    # to `losses`
    losses = []

    # TODO: select the "best attribute" to split on, among `split_candidates`
    # and based on the loss function
    best_attr = None

    # TODO: create a partition, using `partition_by` function and the "best"
    # attribute and update it as `inputs_partitioned`. `inputs_partitioned`
    # should be a `collections.defaultdict` object whose keys are the values of
    # the best attribute, and whose values are observations with best attribute
    # values equal to the said key
    inputs_partitioned = None

    # TODO: update the split candidates by removing the best_attr from it,
    # because we have created a split using this attribute
    new_split_candidates = []

    # TODO: Use recursion and a for-loop-like structure to update the
    # `subtree_dict`, making use of `inputs_partitioned` and
    # `new_split_candidates`
    subtree_dict = dict()

    return (best_attr, subtree_dict)


# -------------------- PREDICTION WITH A REGRESSION TREE ------------------

"""************************************************************************
* function:  predict(tree, to_predict)
*
* arguments:
* 		-tree:  a tree built with the build_tree function
*		-to_predict:  a dictionary of features
*
* return value:  a value indicating a prediction of days_until_funded

* TODO:  Write a recursive function that uses "tree" and the values in the
*		 dictionary "to_predict" to output a predicted value.
************************************************************************"""


def predict(tree, to_predict):
    # TODO
    pass
