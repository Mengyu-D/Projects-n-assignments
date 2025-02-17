import csv
import math
import numpy as np
import collections
#from sklearn.model_selection import train_test_split
import pprint
import pandas as pd
import matplotlib
import plotnine as p9
from fontTools.varLib.models import nonNone


# ------------------- READING IN DATA ------------------------------------
def load_data(for_prediction=False):
    file = "loans_A_labeled.csv"
    if for_prediction:
        file = "loans_B_unlabeled.csv"
    data = []

    with open(file, "rt", encoding="utf8") as f:
        dict_reader = csv.DictReader(f)
        for observation in dict_reader:
            features = {}
            exclude = {"id", "name", "description", "posted_date", "town"}
            for key in observation:
                if key not in exclude and key != "days_until_funded":
                    if key == "languages":
                        langs = observation[key].strip().split("|")
                        langs = [langs.strip() for langs in langs if langs.strip()]
                        all_langs = set()
                        for lang in langs:
                            features[f'lang_{lang}'] = 1
                            all_langs.add(lang)
                        for lang in all_langs:
                            if features[f'lang_{lang}'] not in features:
                                features[f'lang_{lang}'] = 0
                        # features['lang_en'] = 1 if 'en' in langs else 0
                        # features['lang_es'] = 1 if 'es' in langs else 0

                    elif key == "gender":
                        features['gender_F'] = 1 if observation[key] == 'F' else 0
                    else:
                        features[key] = observation[key]
            if not for_prediction:
                target = int(observation["days_until_funded"])
                data.append((features, target))
            else:
                data.append(features)

    if not for_prediction:
        data = continuous_to_percentile(data, "loan_amount", 4)
        data = continuous_to_percentile(data, "repayment_term", 4)
    return data



# ------------------- CREATING ADDITIONAL FEATURES ---------------------------

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

LoanA_labeled = load_data(for_prediction=False)

def get_feature(for_prediction=False):
    file = "loans_A_labeled.csv"
    if for_prediction:
        file = "loans_B_unlabeled.csv"
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
                                    labels=[f'Q{i + 1}' for i in range(n_bins)])

    grouped_loan_amount = (df.groupby("loan_amount_bin")["days_until_funded"]
                           .mean()
                           .reset_index()
                           .sort_values("days_until_funded"))

    return ({"sector": grouped_sector},
            {"activity": grouped_activity},
            {"loan_amount": grouped_loan_amount})
# -------------------- BUILDING BLOCKS FOR TREES -------------------------
# def partition_loss(subsets):
#     num_obs = sum(len(subset) for subset in subsets)
#
#     loss = 0
#     for subset in subsets:
#         counter = collections.Counter(label for _, label in subset)
#         prediction = counter.most_common(1)[0]
#         h = (1 - prediction[1] / float(len(subset))) * (len(subset) / float(num_obs))
#         loss = loss + h
#
#     return loss
def partition_loss(subsets):
    total_loss = 0
    for subset in subsets:
        if not subset:
            continue
        targets = []
        for obs in subset:
            targets.append(obs[1])
        mean = sum(targets) / len(targets)
        subset_loss = sum((t-mean)**2 for t in targets)
        total_loss += subset_loss
    return total_loss


def partition_by(inputs, attribute):
    groups = collections.defaultdict(list)
    for input_ in inputs:
        key = input_[0].get(attribute)
        if key:
            groups[key].append(input_)
    return groups

def partition_loss_by(inputs, attribute):
    partitions = partition_by(inputs, attribute)
    return partition_loss(partitions.values())

def calculate_mse(labels):
    if len(labels) == 0:
        return 0
    mean_label = np.mean(labels)
    return np.mean([(label - mean_label) ** 2 for label in labels])

# -------------------- BUILDING A REGRESSION TREE -------------------------
def build_tree(inputs, num_levels, split_candidates = None):
    if split_candidates == None:
        split_candidates = inputs[0][0].keys()

    if num_levels == 0 or not split_candidates:
        labels = [label for _, label in inputs]
        counter = collections.Counter(labels)
        return counter.most_common(1)[0][0]

    best_attr = min(split_candidates, key=lambda attr:partition_loss_by(inputs, attr))
    partitions = partition_by(inputs, best_attr)
    split_candidates = [attr for attr in split_candidates if attr != best_attr]
    subtrees = {
        value: build_tree(subset, num_levels-1, split_candidates)
        for value, subset in partitions.items()
    }
    return (best_attr, subtrees)

# -------------------- PREDICTION WITH A REGRESSION TREE ------------------
def predict(tree, to_predict, default=None):
    if isinstance(tree, int) or isinstance(tree, float):
        return tree

    if isinstance(tree, tuple) and len(tree) == 2:
        attribute, subtrees = tree
    else:
        return default

    attribute_value = to_predict.get(attribute, None)

    if attribute_value not in subtrees:
        return default

    subtree = subtrees.get(attribute_value, default)
    return predict(subtree, to_predict, default=subtrees.get(default, default))

if __name__ == "__main__":
    training_data = load_data(for_prediction=False)
    decision_tree = build_tree(training_data, 3)

    test_data = load_data(for_prediction=True)
    b_predictions = [predict(decision_tree, features) for features in test_data]
    #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    total_mse = partition_loss_by(training_data, list(training_data[0][0].keys())[0])
    print(f"MSE: {total_mse}")

if __name__ == "__main__":
    file = "loans_A_labeled.csv"
    df = pd.read_csv(file)

    feature_avg = group_feature(df, n_bins=4)
    print(feature_avg)