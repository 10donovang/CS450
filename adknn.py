from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from scipy.io import arff

pd.options.mode.chained_assignment = None  # default='warn'


def run_car():
    info = pd.read_csv("car.txt")
    X = info.loc[:, info.columns != '__class']
    y = info.__class
    X["safety"] = X["safety"].astype('category')
    X["safety"] = X["safety"].cat.codes
    X["lug_boot"] = X["lug_boot"].astype('category')
    X["lug_boot"] = X["lug_boot"].cat.codes
    X["buying"] = X["buying"].astype('category')
    X["buying"] = X["buying"].cat.codes
    X["maint"] = X["maint"].astype('category')
    X["maint"] = X["maint"].cat.codes
    X["doors"] = X["doors"].astype('category')
    X["doors"] = X["doors"].cat.codes
    X["persons"] = X["persons"].astype('category')
    X["persons"] = X["persons"].cat.codes
    return X, y


def run_mpg():
    info = pd.read_csv("mpg.cvs", dtype="object")
    X = info.loc[:, info.columns != 'mpg']
    y = info.loc[:, info.columns == 'mpg']
    print(X)
    return X, y


def run_autism():
    data = arff.loadarff('Autism-Adult-Data Plus Description File/Autism-Adult-Data.arff')
    info = pd.DataFrame(data[0])
    X = info.loc[:, info.columns != 'Class/ASD']
    y = info.loc[:, info.columns == 'Class/ASD']
    X["A1_Score"] = X["A1_Score"].astype('category')
    X["A1_Score"] = X["A1_Score"].cat.codes
    X["A2_Score"] = X["A2_Score"].astype('category')
    X["A2_Score"] = X["A2_Score"].cat.codes
    X["A3_Score"] = X["A3_Score"].astype('category')
    X["A3_Score"] = X["A3_Score"].cat.codes
    X["A4_Score"] = X["A4_Score"].astype('category')
    X["A4_Score"] = X["A4_Score"].cat.codes
    X["A5_Score"] = X["A5_Score"].astype('category')
    X["A5_Score"] = X["A5_Score"].cat.codes
    X["A6_Score"] = X["A6_Score"].astype('category')
    X["A6_Score"] = X["A6_Score"].cat.codes
    X["A7_Score"] = X["A7_Score"].astype('category')
    X["A7_Score"] = X["A7_Score"].cat.codes
    X["A8_Score"] = X["A8_Score"].astype('category')
    X["A8_Score"] = X["A8_Score"].cat.codes
    X["A9_Score"] = X["A9_Score"].astype('category')
    X["A9_Score"] = X["A9_Score"].cat.codes
    X["A10_Score"] = X["A10_Score"].astype('category')
    X["A10_Score"] = X["A10_Score"].cat.codes
    X["gender"] = X["gender"].astype('category')
    X["gender"] = X["gender"].cat.codes
    X["ethnicity"] = X["ethnicity"].astype('category')
    X["ethnicity"] = X["ethnicity"].cat.codes
    X["jundice"] = X["jundice"].astype('category')
    X["jundice"] = X["jundice"].cat.codes
    X["austim"] = X["austim"].astype('category')
    X["austim"] = X["austim"].cat.codes
    X["contry_of_res"] = X["contry_of_res"].astype('category')
    X["contry_of_res"] = X["contry_of_res"].cat.codes
    X["used_app_before"] = X["used_app_before"].astype('category')
    X["used_app_before"] = X["used_app_before"].cat.codes
    X["age_desc"] = X["age_desc"].astype('category')
    X["age_desc"] = X["age_desc"].cat.codes
    X["relation"] = X["relation"].astype('category')
    X["relation"] = X["relation"].cat.codes
    X["age"] = X["age"].astype('category')
    X["age"] = X["age"].cat.codes
    X["result"] = X["result"].astype('category')
    X["result"] = X["result"].cat.codes
    return X, y


def knn(X, y):
    k_range = range(1, 31)
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
        k_scores.append(scores.mean())
    print k_scores


def main():
    X, y = run_car()
    knn(X, y)
    X, y = run_autism()
    knn(X, y)
    #X, y = run_mpg()
    #knn(X, y)
# I can't figure out how to read in the third bit of data correctly. It keeps coming up as NaN when I try and
# split it up. It's really strange.

if __name__ == "__main__":
    main()
