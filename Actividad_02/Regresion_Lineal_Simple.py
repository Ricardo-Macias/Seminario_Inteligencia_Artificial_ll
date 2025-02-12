#Importar paquetes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Leer datos
data = pd.read_csv('A01_Pandas/countries.csv')
df_mex = data[data.country == 'Mexico']
#df_mex.plot.scatter(x='year', y='lifeExp')
#plt.show()

#Seleccionar variables
x = np.asanyarray(df_mex[['year']])
y = np.asanyarray(df_mex[['lifeExp']])

#Crear y entrenar modelos
model = LinearRegression()
model.fit(x,y)

#Graficar el resultado
ypred = model.predict(x)
plt.scatter(x,y)
plt.plot(x, ypred, '--r')
plt.title('Regresion Lineal')
plt.show()

#Metricas de desempeño
import sklearn.metrics as m

print('MAE: ', m.mean_absolute_error(y, ypred))
print('MSE: ', m.mean_squared_error(y, ypred))
print('MedAE: ', m.median_absolute_error(y, ypred))
print('r2-score: ', m.r2_score(y, ypred))
print('EVS: ', m.explained_variance_score(y, ypred))