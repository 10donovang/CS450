import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.model_selection import train_test_split
import scipy as sp
import random
from sklearn import datasets

pd.options.mode.chained_assignment = None  # default='warn'


def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    try:
        entropy = np.sum(
            [(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    except ZeroDivisionError:
        print ("WARNING: Invalid Equation")
    return entropy


def InfoGain(data, split_attribute_name, target_name="yegvx"):

    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    Weighted_Entropy = np.sum(
        [(counts[i] / np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
         for i in range(len(vals))])
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain


def ID3(data, originaldata, features, target_attribute_name="yegvx", parent_node_class=None):
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name])[
            np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]

    elif len(features) == 0:
        return parent_node_class

    else:
        parent_node_class = np.unique(data[target_attribute_name])[
            np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        item_values = [InfoGain(data, feature, target_attribute_name) for feature in
                       features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        tree = {best_feature: {}}
        features = [i for i in features if i != best_feature]

        for value in np.unique(data[best_feature]):
            value = value

            sub_data = data.where(data[best_feature] == value).dropna()

            subtree = ID3(sub_data, data_train, features, target_attribute_name, parent_node_class)

            tree[best_feature][value] = subtree

        return tree


def predict(query, tree, default=1):

    # 1.
    for key in list(query.keys()):
        if key in list(tree.keys()):
            # 2.
            try:
                result = tree[key][query[key]]
            except:
                return default

            # 3.
            result = tree[key][query[key]]
            # 4.
            if isinstance(result, dict):
                return predict(query, result)
            else:
                return result


def test(data, tree):
    # Create new query instances by simply removing the target feature column from the original dataset and
    # convert it to a dictionary
    queries = data.iloc[:, :-1].to_dict(orient="records")

    # Create a empty DataFrame in whose columns the prediction of the tree are stored
    predicted = pd.DataFrame(columns=["predicted"])

    # Calculate the prediction accuracy
    for i in range(len(data)):
        predicted.loc[i, "predicted"] = predict(queries[i], tree, 1.0)
    print('The prediction accuracy is: ', (np.sum(predicted["predicted"] == data["yegvx"]) / len(data)) * 100, '%')


def load_data():
    seed = random.randint(0, 999)

    inputs = pd.read_csv("letters.txt")
    letters = inputs.letter
    new_inputs = inputs.loc[:, inputs.columns != 'letter']

    data_train, data_test, targets_train, targets_test = \
        train_test_split(new_inputs, letters, test_size=0.30, random_state=seed)
    return data_train, data_test


data_train, data_test = load_data()


def main():
    tree = ID3(data_train, data_train, data_train.columns)
    pprint(tree)
    test(data_train, tree)


if __name__ == "__main__":
    main()
