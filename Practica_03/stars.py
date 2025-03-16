import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Leer dataset
stars_dataset = pd.read_csv('Practica_03/Stars.csv')

# Reemplazar
star_category = pd.get_dummies(stars_dataset['Star category'])

star_color = pd.get_dummies(stars_dataset['Star color'])

stars_dataset.replace({'A':0,'B':1,'F':2,'G':3,'K':4,'M':5,'O':6}, inplace=True)

stars_dataset.drop(columns=['Star category',
                            'Star color'], inplace=True)

stars_dataset = pd.concat([stars_dataset, star_category, star_color], axis=1)


# Seleccionar variables
x = np.array(stars_dataset.drop(columns=['Spectral Class']))
y = np.array(stars_dataset[['Spectral Class']])

# Escalar los datos
x = StandardScaler().fit_transform(x)

# Crear y entrenar modelos
xtrain, xtest, ytrain, ytest = train_test_split(x, y.ravel(), test_size=0.5)

model = KNeighborsClassifier(2)
model.fit(xtrain, ytrain.ravel())

print('Train: ',model.score(xtrain, ytrain.ravel()))
print('Test: ', model.score(xtest, ytest.ravel()))
