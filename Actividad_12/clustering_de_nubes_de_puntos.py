import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, mixture

plt.close('all')

pc = np.load('Actividad_12/armadillo.npy').T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pc[:,0], pc[:,1],pc[:,2],s=0.1)
plt.show()

#model = cluster.KMeans(n_clusters=5)
model = mixture.GaussianMixture(n_components=20, covariance_type='full')
model.fit(pc)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pc[:,0],pc[:,1],pc[:,2], s=0.1, c=model.predict(pc))
plt.title('Segmentacion')
plt.show()