import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

df = pd.read_csv('Actividad_07/diabetes.csv')

#pd.plotting.scatter_matrix(df)
#corr = df.corr()
#import seaborn as sns
#sns.heatmap(corr,
#            xticklabels=corr.columns,
#            yticklabels=corr.columns)

x = np.asanyarray(df.drop(columns=['Outcome']))
y = np.asanyarray(df[['Outcome']])

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.5)

model = Pipeline([('scaler', StandardScaler()),
                  ('log_reg', LogisticRegression())])

model.fit(xtrain, ytrain.ravel())

print('Train: ', model.score(xtrain, ytrain.ravel()))
print('Test: ', model.score(xtest, ytest.ravel()))

coef = np.abs(model.named_steps['log_reg'].coef_[0])
coef = coef / np.sum(coef)
labels = (df.columns[:-1])
Exp = pd.DataFrame()
Exp['Features'] = labels
Exp['Importance'] = coef
Exp.sort_values(by='Importance')
Exp.set_index('Features', inplace=True)
Exp.Importance.plot(kind='pie')
plt.ylabel('')
plt.title('Explicacion de Variables')
plt.show()
