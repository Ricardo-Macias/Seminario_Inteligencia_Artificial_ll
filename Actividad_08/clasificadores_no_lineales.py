import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

# Instanciar modelos
models = {'KNN':KNeighborsClassifier(2),
          'SVM':SVC(gamma=2, C=1),
          'GP':GaussianProcessClassifier(1.0*RBF(1.0)),
          'DT':DecisionTreeClassifier(max_depth=5),
          'MLP':MLPClassifier(alpha=1, max_iter=1000),
          'NB':GaussianNB()}

# Crear datos
x, y = make_classification(n_features=2, n_redundant=0,
                           n_informative=2, n_clusters_per_class=1)

x += 1 * np.random.rand(*x.shape)
lin_separable = (x, y)

datasets = [make_moons(noise=0.1),
            make_circles(noise=0.1, factor=0.5),
            lin_separable]

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

# Cambiar esto
model_name = 'NB'

for ds_cnt, ds in enumerate(datasets):
    # Leer datasets
    x, y = ds
    x = StandardScaler().fit_transform(x)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y)

    # Crear dominio de dibujo ---------------------------
    xmin, xmax = x[:,0].min() - 0.5, x[:,0].max() + 0.5
    ymin, ymax = x[:,1].min() - 0.5, x[:,1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 100),
                         np.linspace(ymin, ymax, 100))
    #----------------------------------------------------

    model = models[model_name]
    model.fit(xtrain, ytrain)
    score_train = model.score(xtrain, ytrain)
    score_test = model.score(xtest, ytest)

    # Dibujar -------------------------------------------

    ax = plt.subplot(1, 3, ds_cnt+1)
    if hasattr(model, 'decision_function'):
        zz = model.decision_function(np.c_[xx.ravel(),
                                           yy.ravel()])[:,1]
    
    else:
        zz = model.predict_proba(np.c_[xx.ravel(),
                                       yy.ravel()])[:,1]
        
    zz = zz.reshape(xx.shape)
    ax.contourf(xx, yy, zz, cmap=cm, alpha=0.8)

    ax.scatter(xtrain[:,0], xtrain[:,1], c=ytrain,
               cmap=cm_bright, edgecolors='k')
    
    ax.scatter(xtest[:,0], xtest[:,1], c=ytest,
               cmap=cm_bright, edgecolors='k', alpha=0.6)
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(())
    ax.set_yticks(())

    ax.text(xmax-0.3, ymin+0.7, '%.2f'%score_train,
            size=15, horizontalalignment='right')
    
    ax.text(xmax-0.3, ymin+0.3, '%.2f'%score_test,
            size=15, horizontalalignment='right')
    
plt.tight_layout()
plt.suptitle('DT')
plt.show()