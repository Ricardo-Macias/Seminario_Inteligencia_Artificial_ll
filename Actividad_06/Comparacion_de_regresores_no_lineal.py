import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor

# Crear datos sinteticos
np.random.seed(42)
m = 300 # Muestras
r = 0.5 #Ruido
x = 6 * np.random.rand(m, 1) - 3
y = 0.5 * x**2 + x + 2 + r*np.random.randn(m, 1)
xtrain, xtest, ytrain, ytest = train_test_split(x, y)

# Crear modelo y entrenamiento
"""
# ARBOL DE DECISIONES
model = DecisionTreeRegressor(max_depth=3)
plt.title('Arbol de decision')
"""
"""
# K-VECINOS MAS CERCANOS
model = KNeighborsRegressor(n_neighbors=1)
plt.title('K-Vecinos mas cercanos')
"""
"""
# SVM
model = SVR(kernel='rbf', gamma=200)
plt.title('SVM')
"""
"""
# KERNEL RIDGE
model = KernelRidge(kernel='rbf', alpha=1)
plt.title('Kernel Ridge')
"""
# MLP
model = MLPRegressor(hidden_layer_sizes=(1000,))
plt.title('MLP')
model.fit(xtrain, ytrain)
print('R2 Train: ', model.score(xtrain, ytrain))
print('R2 Test: ', model.score(xtest, ytest))

# Dibujar experimento
xnew = np.linspace(-3, 3, 400).reshape(-1, 1)
ynew = model.predict(xnew)
plt.plot(xnew, ynew, '-k', linewidth=3.5)
plt.plot(xtrain, ytrain, '.b')
plt.plot(xtest, ytest, '.r')
plt.xlabel('$x$', fontsize=18)
plt.ylabel('$y$', fontsize=18, rotation=0)
plt.axis([-3,3,0,10])
plt.show()