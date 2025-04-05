# %% Bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# %% Leer datos
data = pd.read_csv('Actividad_13/mnist_784.csv')
n_samples = 10000
x = np.asanyarray(data.drop(columns=['class']))[:n_samples,:]
y = np.asanyarray(data[['class']])[:n_samples].ravel()
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.1)

# %% Dibujar muestras aleatorias
sample = np.random.randint(n_samples)
plt.imshow(x[sample].reshape(28,28),cmap=plt.cm.gray)
plt.title('Target: %i' % y[sample])
plt.show()

# %% Crear y entrenar modelo
model = Pipeline([('scaler', StandardScaler()),
                  ('PCS', PCA(n_components=50)),
                  ('svm', svm.SVC(gamma=0.0001))])
model.fit(xtrain, ytrain)

# ½½ Metricas de desempeño
print('Train: ', model.score(xtrain, ytrain))
print('Test: ', model.score(xtest, ytest))
ypred = model.predict(xtest)
print('classification report \n',
      metrics.classification_report(ytest, ypred))
print('Condusion matriz \n',
      metrics.confusion_matrix(ytest, ypred))

# %% Dibujar ejemplo
sample = np.random.randint(xtest.shape[0])
plt.imshow(xtest[sample].reshape(28,28), cmap=plt.cm.gray)
plt.title('Prediction: %i' % ypred[sample])
plt.show()
