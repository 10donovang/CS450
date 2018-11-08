import pandas as pd
import numpy as np
from pprint import pprint
#from sklearn.model_selection import train_test_split
import scipy as sp
import random
from sklearn import tree
import graphviz

pd.options.mode.chained_assignment = None  # default='warn'


def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    try:
        entropy = np.sum(
            [(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    except ZeroDivisionError:
        print ("WARNING: Invalid Equation")
    return entropy


def InfoGain(data, split_attribute_name, target_name="Black_King_rank"):

    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    Weighted_Entropy = np.sum(
        [(counts[i] / np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
         for i in range(len(vals))])
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain


def ID3(data, originaldata, features, target_attribute_name="Black_King_rank", parent_node_class=None):
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


def train_test_split(dataset):
    training_data = dataset.iloc[:80].reset_index(drop=True)
    testing_data = dataset.iloc[80:].reset_index(drop=True)
    return training_data, testing_data


def test(data, tree):
    queries = data.iloc[:, :-1].to_dict(orient="records")

    predicted = pd.DataFrame(columns=["predicted"])

    # Calculate the prediction accuracy
    for i in range(len(data)):
        predicted.loc[i, "predicted"] = predict(queries[i], tree, 1.0)
    print('The prediction accuracy is: ', (np.sum(predicted["predicted"] == data["Black_King_rank"]) / len(data)) * 100, '%')


def load_data():
    dataset = pd.read_csv('chess.txt',
                          names=['White_King_file', 'White_King_rank', 'White_Rook_file', 'White_Rook_rank',
                                 'Black_King_file',
                                 'Black_King_rank', 'Turns_to_win', ])
    dataset = dataset.drop('Turns_to_win', axis=1)
    subs = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h':8}
    dataset.White_King_file = [subs.get(item, item) for item in dataset.White_King_file]
    dataset.Black_King_file = [subs.get(item, item) for item in dataset.Black_King_file]
    dataset.White_Rook_file = [subs.get(item, item) for item in dataset.White_Rook_file]
    data_train, data_test = train_test_split(dataset)
    return data_train, data_test


def load_data_sk():
    dataset = pd.read_csv('chess.txt',
                          names=['White_King_file', 'White_King_rank', 'White_Rook_file', 'White_Rook_rank',
                                 'Black_King_file',
                                 'Black_King_rank', 'Turns_to_win', ])
    subs = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8}
    dataset.White_King_file = [subs.get(item, item) for item in dataset.White_King_file]
    dataset.Black_King_file = [subs.get(item, item) for item in dataset.Black_King_file]
    dataset.White_Rook_file = [subs.get(item, item) for item in dataset.White_Rook_file]
    data_train, data_test = train_test_split(dataset)
    return data_train, data_test


data_train, data_test = load_data()


def main():
    my_tree = ID3(data_train, data_train, data_train.columns[:-1])
    pprint(my_tree)
    test(data_train, my_tree)
    train, testD = load_data_sk()
    turns = train.Turns_to_win
    data = train.loc[:, train.columns != 'Turns_to_win']
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(data, turns)
    dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=['White_King_file', 'White_King_rank', 'White_Rook_file', 'White_Rook_rank',
                                 'Black_King_file',
                                 'Black_King_rank'],
                         class_names='Turns_to_win',
                         filled=True, rounded=True,
                         special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render("Chess")


if __name__ == "__main__":
    main()
