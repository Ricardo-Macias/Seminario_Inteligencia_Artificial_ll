import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Actividad_05/daily-min-temperatures.csv')

x = np.asanyarray(data['Temp'])
plt.plot(x)
plt.title('Serie de Tiempo')
plt.xlabel('Fecha')
plt.ylabel('Temperatura °C')

plt.figure()
p = 7
plt.scatter(x[p:], x[:-p])
plt.xlabel('$x_t$')
plt.ylabel('$x_[t-p$')
print('Correlacion: ', np.corrcoef(x[p:].T, x[:-p].T))

plt.figure()
pd.plotting.autocorrelation_plot(data.Temp)
plt.show()