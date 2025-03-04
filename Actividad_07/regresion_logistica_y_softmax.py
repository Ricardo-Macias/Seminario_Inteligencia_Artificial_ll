# Paqueteria
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Leer datos
iris = load_iris()

x = iris['data']
y = iris['target']

xtrain, xtest, ytrain, ytest = train_test_split(x, y)

sofmax = LogisticRegression(multi_class='multinomial')
sofmax.fit(xtrain, ytrain)

print('F1 Train: ', sofmax.score(xtrain, ytrain))
print('F1 Test: ', sofmax.score(xtest, ytest))

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

ypred =sofmax.predic(xtest)

cm = confusion_matrix(ytest, ypred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

print('Reporte de Clasificacion: ')
print(classification_report(ytest, ypred))