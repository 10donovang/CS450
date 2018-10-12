from sklearn import datasets
from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# from sklearn.linear_model import LogisticRegression

# from yellowbrick.classifier import ConfusionMatrix

import numpy as np
import scipy as sp
import random


class HardCodedClassifier:
    def __init__(self):
        self.results_list = []

    def fit(self, data_train, targets_train):
        pass

    def predict(self, data_test):
        for x in data_test:
            self.results_list.append(0)

    def results(self, targets_test):
        percent = 0
        x = 0
        while x < len(self.results_list):
            if self.results_list[x] == 0:
                print ("Setosa")
            elif self.results_list[x] == 1:
                print ("Versicolor")
            elif self.results_list[x] == 2:
                print ("Virginica")
            else:
                print ("mistake")
            if self.results_list[x] == targets_test[x]:
                percent += 1
            x += 1

        percent = float(100 * float(percent) / len(self.results_list))

        print (str(round(percent, 2)) + "%")

#        model = LogisticRegression()

        # The ConfusionMatrix visualizer taxes a model
#        cm = ConfusionMatrix(model, classes=[0, 1, 2])

        # To create the ConfusionMatrix, we need some test data. Score runs predict() on the data
        # and then creates the confusion_matrix from scikit-learn.
#        cm.score(self.results_list, targets_test)

        # How did we do?
#        cm.poof()


class knn:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors


    def fit(self, data_train, targets_train):
        n_samples = data_train.shape[0]

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


def main():

    seed = random.randint(0, 999)

    iris = datasets.load_iris()

    data_train, data_test, targets_train, targets_test = \
        train_test_split(iris.data, iris.target, test_size=0.30, random_state=seed)

#    classifier = GaussianNB()
#    model = classifier.fit(data_train, targets_train)

#    targets_predicted = model.predict(data_test)

#    my_classifier = HardCodedClassifier()

#    my_classifier.fit(data_train, targets_train)

#    my_classifier.predict(data_test)

#    my_classifier.results(targets_test)

    classifier = KNeighborsClassifier(n_neighbors=11)
    model = classifier.fit(data_train, targets_train)
    predictions = model.predict(data_test)

    my_classifier = knn(11)
    my_classifier.fit(data_train, targets_train)
    my_predictions = my_classifier.predict(data_test)

    accuracy = accuracy_score(targets_test, predictions) * 100
    print('Accuracy of default model is equal ' + str(round(accuracy, 2)) + ' %.')

    my_accuracy = accuracy_score(targets_test, my_predictions) * 100
    print('Accuracy of our model is equal ' + str(round(my_accuracy, 2)) + ' %.')


if __name__ == "__main__":
    main()
