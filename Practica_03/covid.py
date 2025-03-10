import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

covid_dataset = pd.read_csv('Practica_03/Covid Dataset.csv')
#covid_dataset.replace(np.nan, 'no_info', inplace=True)

# Elimina la columna No y las columnas Wearing Masks y Sanitization from Market
new_dataset = pd.get_dummies(covid_dataset, 
                                    drop_first=True)

# Limpia


x = np.array(new_dataset.drop(columns=['COVID-19_Yes']))
y = np.array(new_dataset[['COVID-19_Yes']])

x = StandardScaler().fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.5)

model = MLPClassifier(alpha=1, max_iter=1000)
model.fit(xtrain, ytrain.ravel())

print('Train: ',model.score(xtrain, ytrain.ravel()))
print('Test: ', model.score(xtest, ytest.ravel()))
