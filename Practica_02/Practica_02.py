import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

models = {'DT':DecisionTreeClassifier(max_depth=7),
          'KNN':KNeighborsClassifier(3),
          'MLP':MLPClassifier(alpha=1, max_iter=1000),
          'NB':GaussianNB()}

dataset = pd.read_csv('Practica_02/loan_prediction.csv')

x = np.array(dataset.drop(columns=['Loan_Status']))
y = np.array(dataset[['Load_Status']])

x = StandardScaler().fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.5)

score_train = []
score_test = []

for count in models:
    model = models[count]
    model.fit(xtrain, ytrain)

    score_train.append(model.score(xtrain, ytrain))
    score_test.append(model.score(xtest, ytest))

