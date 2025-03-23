import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

def kmeans_silhoette(data, n_clusters):
    plt.figure()
    plt.xlim([-0.2,1])

    model = KMeans(n_clusters=n_clusters)
    labels =model.fit_predict(data)

    s_avg = silhouette_score(data, labels)
    print('For ' + str(n_clusters) + 'clusters')
    print('silhoutte score: ', s_avg)

    samples_sil_values = silhouette_samples(data, labels)
    y_lower = 10

    for i in range(n_clusters):
        # Extraer coheficientes pro cluster y rango para dibujar
        icluster_sv = samples_sil_values[labels == i]
        icluster_sv.sort()

        # Obtener tamaño del cluster y rango para dibujar
        size_icluster = icluster_sv.shape[0]
        y_upper = y_lower + size_icluster

        # Obtener color del cluster
        color = plt.cm.nipy_spectral(float(i)/n_clusters)

        # Dibujar barras
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, icluster_sv, facecolor=color,
                          edgecolor=color, alpha=0.7)
        plt.text(-0.5, y_lower+0.5 * size_icluster, str(i))

        y_lower = y_upper + 10
    
    plt.title('Coheficiente de silueta')
    plt.xlabel('Valores del coheficiente de silueta')
    plt.ylabel('Agrupaciones')

    plt.axvline(x=s_avg, color='red', linestyle='--')
    plt.yticks([])
    plt.xticks(np.arange(-0.1,1.1,0.1))
    plt.show()


def elbow(x, max_clusters=10):
    result = np.zeros(max_clusters)
    for i in range(2, max_clusters+1):
        model = KMeans(n_clusters=i)
        labels = model.fit_predict(x)
        result[i-1] = silhouette_score(x, labels)
    
    plt.figure()
    plt.plot(result)
    plt.title('Grafica del codo')
    plt.show()

data = pd.read_csv('Actividad_10/Mall_Customers.csv')
data.replace({'Male':0, 'Female':1}, inplace=True)
data.drop(columns=['CustomerID'], inplace=True)

x = np.asanyarray(data)

kmeans_silhoette(x, 6)
elbow(x)