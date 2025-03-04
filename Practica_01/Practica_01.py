import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv('Practica_01/weatherHistory.csv')

dataset.replace(np.nan, 'no_info', inplace=True)

precip = pd.get_dummies(dataset['Precip Type'])
summary = pd.get_dummies(dataset['Summary'])

dataset.drop(columns=['Precip Type', 
                      'Daily Summary', 
                      'Summary', 
                      'Formatted Date'], inplace=True)

dataset = pd.concat([dataset, precip, summary], axis=1)

x = np.array(dataset.drop(columns=['Visibility (km)']))
y = np.array(dataset[['Visibility (km)']])

x = StandardScaler().fit_transform(x)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.5)

# Crear modelo
model = DecisionTreeRegressor(max_depth=4)
model.fit(xtrain, ytrain)
print('Train: ', model.score(xtrain, ytrain))
print('Test: ', model.score(xtest, ytest))

