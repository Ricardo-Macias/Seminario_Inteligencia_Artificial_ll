#Importa bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Leer datos
df = pd.read_csv('Actividad_03/home_data.csv')
df2 = df.drop(columns=['id', 'date', 'price', 'zipcode'])

#Seleccionar variables
x = np.asanyarray(df2)
y = np.asanyarray(df[['price']])

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

#Escalar datos
x = StandardScaler().fit_transform(x)

#Entrenar la IA
model = LinearRegression()
model.fit(x,y)

#Calcular metricas
from sklearn.metrics import r2_score, mean_absolute_error

ypred = model.predict(x)
print('R2: ', r2_score(y, ypred))
print('MAE: ', mean_absolute_error(y, ypred))
print('R2: ', model.score(x,y))

#Explicacion de variables
coef = np.abs(model.coef_)
coef = coef / np.sum(coef)
labels = list(df2.columns)
Exp = pd.DataFrame()
Exp['Features'] = labels
Exp['Importance'] = coef.reshape(-1,1)
Exp.sort_values('Importance', inplace=True, ascending=False)
Exp.set_index('Features', inplace=True)
Exp.Importance.plot(kind='pie')
plt.ylabel('')
plt.title('Explicacion de variables: \n Precio de una casa')
plt.show()