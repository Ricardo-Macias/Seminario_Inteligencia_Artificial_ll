import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Actividad_05/daily-min-temperatures.csv')

# Tabular datos
p = 5
data2 = pd.DataFrame(data.Temp)
for i in range(1, p+1):
    data2 = pd.concat([data2, data.Temp.shift(-i)], axis=1)

data2 = data2[:-p]

x = np.asanyarray(data2.iloc[:,:-1])
y = np.asanyarray(data2.iloc[:,-1])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.15)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtrain, ytrain)

print('Train: ', model.score(xtrain, ytrain))
print('Test: ', model.score(xtest, ytest))