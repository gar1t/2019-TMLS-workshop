#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:25:14 2017

@author: chadliamino
"""

import warnings

import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

from sklearn import metrics

random_seed = 1
test_split = 0.3
model = "nb"
data = 'kddcup99.csv'
normal = "normal"

if model == "nb":
    print("Using Gaussian NB classifier")
    classifier = GaussianNB()
elif model == "tree":
    print("Using decision tree classifier")
    classifier = DecisionTreeClassifier()
elif model == "forest":
    print("Using random forest classifier")
    classifier = RandomForestClassifier()
else:
    assert False, model

print("Loading data")
with open(data) as f:
    f.readline()
    dataset = pd.read_csv(f, header=None)
cols = dataset.columns.tolist()
dataset[cols[-1]] = dataset[cols[-1]].apply(
    lambda x: "normal" if x.startswith("normal") else "attack")

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 41].values

print("Encoding data")

labelencoder_x_1 = LabelEncoder()
labelencoder_x_2 = LabelEncoder()
labelencoder_x_3 = LabelEncoder()

x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])
x[:, 3] = labelencoder_x_3.fit_transform(x[:, 3])

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=Warning)
    onehotencoder_1 = OneHotEncoder(categorical_features=[1])
    x = onehotencoder_1.fit_transform(x).toarray()

    onehotencoder_2 = OneHotEncoder(categorical_features=[4])
    x = onehotencoder_2.fit_transform(x).toarray()

    onehotencoder_3 = OneHotEncoder(categorical_features=[70])
    x = onehotencoder_3.fit_transform(x).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=test_split,
    random_state=random_seed)

sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

print("Cross validating model")

scores = cross_validate(
    estimator=classifier,
    X=x_train,
    y=y_train,
    cv=5,
    scoring=["accuracy", "precision", "recall", "f1"])

print("Accuracy: %f" % scores["test_accuracy"].mean())
print("Precision: %f" % scores["test_precision"].mean())
print("Recall: %f" % scores["test_recall"].mean())
print("F-1: %f" % scores["test_f1"].mean())
