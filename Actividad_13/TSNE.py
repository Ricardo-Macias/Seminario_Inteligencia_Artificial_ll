# %% Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# %% Leer datos
data = pd.read_csv('Actividad_13/mnist_784.csv')
n_samples = 5000
x = np.asanyarray(data.drop(columns=['class']))[:n_samples,:]
y = np.asanyarray(data[['class']])[:n_samples].ravel()

# %% Crear modelo y entrenar
model = TSNE(n_components=2, n_iter=2000)
x_2d = model.fit_transform(x)

# %% Dibujar
plt.scatter(x_2d[:,0],x_2d[:,1], c=y, cmap=plt.cm.tab10)
plt.show();
