# %% Importar biblioteca
import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing, tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

# %% Leer datos
df = pd.read_csv('Practica_02/loan_prediction.csv')
x = np.asanyarray(df.iloc[:,:-1])
y = np.asanyarray(df.iloc[:,-1])
xtrain, xtest, ytrain, ytest = train_test_split(x, y)

# %% Entrenar un solo modelo
dt = tree.DecisionTreeClassifier(max_depth=5)
dt.fit(xtrain, ytrain)
print('Train: ', dt.score(xtrain, ytrain))
print('Test: ', dt.score(xtest, ytest))

# %% Entrenar con cross-validation
dt = tree.DecisionTreeClassifier(max_depth=5)
scores = cross_val_score(dt, xtrain, ytrain, cv=5, scoring='f1_macro')
print('Score: ', scores)
print('Validation: ', np.mean(scores))

# %% Definir espacio de hiperparametros
hp = {'max_depth':[1,5,7,10,15],
      'min_samples_leaf':[1,2,5,7,10],
      'min_samples_split':[2, 5, 10, 20],
      'criterion':['gini', 'entropy']}
dt = tree.DecisionTreeClassifier()
search = GridSearchCV(dt, hp, cv=5, scoring='f1_macro')
search.fit(xtrain, ytrain)

print(search.cv_results_['mean_test_score'])
print(search.best_params_)
best_model = search.best_estimator_

# %% RandomSearch
dt = tree.DecisionTreeClassifier()
search = RandomizedSearchCV(dt, hp, cv=5, scoring='f1_macro', n_iter=20)
search.fit(xtrain, ytrain)
print(search.cv_results_['mean_test_score'])
print(search.best_params_)
best_model = search.best_estimator_

# %% Reentrenar modelo
best_model.fit(xtrain, ytrain)
print('Train: ', best_model.score(xtrain, ytrain))
print('Test: ', best_model.score(xtest, ytest))
