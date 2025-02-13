# Import bibliotecas
import numpy as np
import matplotlib.pyplot as plt

# Crear datos sinteticos
np.random.seed(42)
m = 100 # Muestras
r = 1 # Ruido
x = 6 * np.random.rand(m, 1) - 3
y = 0.5 * x**2 + x + r + np.random.randn(m, 1)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25)

# Crear modelo
model = Pipeline([('poly', PolynomialFeatures(degree=3, include_bias=False)),
                  ('scaler', StandardScaler()),
                  ('lin_reg', LinearRegression())])

model.fit(xtrain, ytrain)

print('Train: ', model.score(xtrain, ytrain))
print('Test: ', model.score(xtest, ytest))

# Dibujar
xnew = np.linspace(-3, 3, 100).reshape(-1, 1)
ypred = model.predict(xnew)

plt.plot(xnew, ypred, '--k')
plt.plot(xtrain, ytrain, '.b')
plt.plot(xtest, ytest, '.r')
plt.xlabel('$x_1$', fontsize=18)
plt.ylabel('$y$', fontsize=18, rotation=0)
plt.axis([-3,3,0,10])
plt.title('Regresion Polinomial')
plt.show()