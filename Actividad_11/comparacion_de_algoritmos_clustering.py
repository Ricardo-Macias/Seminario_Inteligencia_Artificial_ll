import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture, metrics
from sklearn.preprocessing import StandardScaler

# Crear datos sinteticos
np.random.seed(0)
n_samples = 500
X = 6 * [None]

# Circulos concentrados
xtemp, _ = datasets.make_circles(n_samples=n_samples,
                                 factor=0.5,
                                 noise=0.05)
X[0] = StandardScaler().fit_transform(xtemp)

# Medias lunas
xtemp, _ = datasets.make_moons(n_samples=n_samples,
                                 noise=0.05)
X[1] = StandardScaler().fit_transform(xtemp)

# Blobs normales
xtemp, _ = datasets.make_blobs(n_samples=n_samples,
                                 random_state=8)
X[2] = StandardScaler().fit_transform(xtemp)

# Plano sin agrupacion
xtemp = np.random.rand(n_samples,2)
X[3] = StandardScaler().fit_transform(xtemp)

# Blobs con deformacion anisotropica
xtemp, _ = datasets.make_blobs(n_samples=n_samples,
                                 random_state=170)
xtemp = np.dot(xtemp, [[0.6, -0.6],[-0.4, 0.8]])
X[4] = StandardScaler().fit_transform(xtemp)

# Blods con varias varianzas
xtemp, _ = datasets.make_blobs(n_samples=n_samples,
                                 random_state=75,
                                 cluster_std=[10.0,2.5,0.5])
X[5] = StandardScaler().fit_transform(xtemp)

clusters = [2,2,3,3,3,3]
eps = [0.3,0.3,0.3,0.3,0.15,0.18]

# ---------------------------------------------------------------------------------------
ypred = []
for e,x in zip(eps, X): # for c, x in (zip(clusters, X)):
    #model = cluster.KMeans(n_clusters=c)
    #model = cluster.Birch(n_clusters=c)
    #model = cluster.SpectralClustering(n_clusters=c, affinity='nearest_neighborn')
    #model = mixture.GaussianMixture(n_clusters=c, cavariance_type='full)
    #model = cluster.DBSCAN(eps=e)
    model = cluster.OPTICS(min_samples=20, xi=0.05, min_cluster_size=0.1)

    model.fit(x)
    if hasattr(model, 'labels_'):
        ypred.append(model.labels_.astype(int))
    else:
        ypred.append(model.fit_predict(x))

fig = plt.figure(figsize=(27,9))
fig.suptitle('OPTICS', fontsize=48)
for i in range(6):
    ax = plt.subplot(2,3,i+1)
    ax.scatter(X[i][:,0], X[i][:,1], c=ypred[i])
plt.show()