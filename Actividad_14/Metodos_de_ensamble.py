import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

# %% Leer y preparar datos
df = pd.read_csv('Actividad_07/diabetes.csv')
x = np.asanyarray(df.iloc[:,:-1])
y = np.asanyarray(df.iloc[:,-1])
xtrain, xtest, ytrain, ytest = train_test_split(x, y)

# %% Un solo modelo
dt = DecisionTreeClassifier(max_depth=5)
dt.fit(xtrain, ytrain)
print('DT train: ', dt.score(xtrain, ytrain))
print('DT test: ', dt.score(xtest, ytest))

# %% Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=5)
rf.fit(xtrain, ytrain)
print('RF train: ', rf.score(xtrain, ytrain))
print('RF test: ', rf.score(xtest, ytest))

# %% Bagging
bg = BaggingClassifier(DecisionTreeClassifier(), n_estimators=100,
                       max_samples=0.5,
                       max_features=1.0)
bg.fit(xtrain, ytrain)
print('BG train: ', bg.score(xtrain, ytrain))
print('BG test: ', bg.score(xtest, ytest))

# %% AdaBoots
ab = AdaBoostClassifier(LogisticRegression(),
                        n_estimators=5,
                        learning_rate=1)
ab.fit(xtrain, ytrain)
print('AB train: ',ab.score(xtrain, ytrain))
print('AB test: ', ab.score(xtest, ytest))

# %% Voting
lr = LogisticRegression(solver='lbfgs', max_iter=500)
svm = SVC(kernel='rbf', gamma=0.005)
dt2 = DecisionTreeClassifier()
vc = VotingClassifier(estimators=[('lr',lr),
                                  ('dt',dt2),
                                  ('svm',svm)],
                      voting='hard')
vc.fit(xtrain, ytrain)
print('VC train: ', vc.score(xtrain, ytrain))
print('VC test: ', vc.score(xtest, ytest))
