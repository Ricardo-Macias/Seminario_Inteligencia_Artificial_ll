# Importar bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpi
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

I = mpi.imread('Actividad_09/imagen.jpg')
I = np.array(I, dtype=np.float64) / 255
plt.imshow(I)
plt.title('Imagen original')

w, h, d = I.shape
n_clusters = 8

Iarray = np.reshape(I, (w*h, d))
x = shuffle(Iarray, random_state=42)[:2000]
model = KMeans(n_clusters=n_clusters)
model.fit(x)

C = model.cluster_centers_

# Dibujo
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:,0],x[:,1],x[:,2], c=x)
ax.set_title('Espacio de color RGB')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:,0],x[:,1],x[:,2], c=model.predict(x))
ax.plot3D(C[:,0,],C[:,1],C[:,2],'+r', linewidth=20)
ax.set_title('Agrupacion de Kmeans')

# Imagen cuantizada
ypred = model.predict(Iarray)
Ilabel = np.reshape(ypred, (w,h))
Iout = np.zeros((w,h,d))
for i in range(w):
    for j in range(h):
        Iout[i,j,:] = C[Ilabel[i,j]]

plt.figure()
plt.imshow(Iout)
plt.title('Imagen cuantizada')
plt.show()
