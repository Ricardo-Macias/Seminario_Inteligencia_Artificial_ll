import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Leer dataset
insurance_dataset = pd.read_csv('Practica_03/insurance.csv')

# Reemplazar
insurance_dataset.replace({'male':0, 'female':1, 'yes':1, 'no':0}, inplace=True)
new_dataset = pd.get_dummies(insurance_dataset['region'])

insurance_dataset.drop(columns=['region'], inplace=True)

insurance_dataset = pd.concat([insurance_dataset,new_dataset], axis=1)

# Seleccionar variables
x = np.array(insurance_dataset.drop(columns=['charges']))
y = np.array(insurance_dataset[['charges']])

# Escalar los datos
x = StandardScaler().fit_transform(x)

# Crear y entrenar modelo
xtrain, xtest, ytrain, ytest = train_test_split(x, y.ravel(), test_size=0.5)

model = DecisionTreeRegressor(max_depth=4)
model.fit(xtrain, ytrain.ravel())

print('Train: ', model.score(xtrain, ytrain))
print('Test: ', model.score(xtest, ytest))