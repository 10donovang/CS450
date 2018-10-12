import numpy as np
import scipy as sp
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class knn:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, data_train, targets_train):
        n_samples = data_train.shape[0]

        print(self.n_neighbors, n_samples)
        if self.n_neighbors > n_samples:
            raise ValueError("Number of neighbors can't be larger then number of samples in training set.")

        if data_train.shape[0] != targets_train.shape[0]:
            raise ValueError("Number of samples in X and y need to be equal.")

        self.classes_ = np.unique(targets_train)

        self.data_train = data_train
        self.targets_train = targets_train

    def predict(self, data_test):
        n_predictions, n_features = data_test.shape

        predictions = np.empty(n_predictions, dtype=int)

        for i in range(n_predictions):

            predictions[i] = single_prediction(self.data_train, self.targets_train, data_test[i, :], self.n_neighbors)

        return predictions


def single_prediction(data_train, targets_train, train, k):
    # number of samples inside training set
    n_samples = data_train.shape[0]

    # create array for distances and targets
    distances = np.empty(n_samples, dtype=np.float64)

    # distance calculation
    for i in range(n_samples):
        distances[i] = (train - data_train[i]).dot(train - data_train[i])

    distances = sp.c_[distances, targets_train]

    sorted_distances = distances[distances[:, 0].argsort()]

    targets = sorted_distances[0:k, 1]

    unique, counts = np.unique(targets, return_counts=True)
    return unique[np.argmax(counts)]


def load_data_car():
    info = pd.read_csv("car.txt")
    inputs = info.loc[:, info.columns != '__class']
    outputs = info.__class
    num_inputs = len(inputs)
    inputs["safety"] = inputs["safety"].astype('category')
    inputs["safety"] = inputs["safety"].cat.codes
    inputs["lug_boot"] = inputs["lug_boot"].astype('category')
    inputs["lug_boot"] = inputs["lug_boot"].cat.codes
    inputs["buying"] = inputs["buying"].astype('category')
    inputs["buying"] = inputs["buying"].cat.codes
    inputs["maint"] = inputs["maint"].astype('category')
    inputs["maint"] = inputs["maint"].cat.codes
    inputs["doors"] = inputs["doors"].astype('category')
    inputs["doors"] = inputs["doors"].cat.codes
    inputs["persons"] = inputs["persons"].astype('category')
    inputs["persons"] = inputs["persons"].cat.codes
    inputs.head()
    return inputs, outputs, num_inputs


def set_up(inputs, outputs):
    seed = random.randint(0, 999)
    data_train, data_test, targets_train, targets_test = \
        train_test_split(inputs, outputs, test_size=0.30, random_state=seed)
    return data_train, data_test, targets_train, targets_test


def predictions(data_train, data_test,  targets_train, targets_test, k):

    my_classifier = knn(k)
    my_classifier.fit(data_train, targets_train)
    #my_predictions = my_classifier.predict(data_test)
     #my_accuracy = accuracy_score(targets_test, my_predictions) * 100
    return #my_accuracy


def main():
    inputs, outputs, num_inputs = load_data_car()
    data_train, data_test, targets_train, targets_test = set_up(inputs, outputs)
    for k in range(1, num_inputs+1):
        accuracy = predictions(data_train, data_test, targets_train, targets_test, outputs, k)
        pass
    accuracy = accuracy/num_inputs
    print(accuracy)


if __name__ == "__main__":
    main()
