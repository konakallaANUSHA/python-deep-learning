import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


df_train = pd.read_csv('train.csv')
df1 = pd.concat([df_train['SalePrice'], df_train['GarageArea']], axis=1)


mean = np.mean(df1)
print(mean)

sd= np.std(df1)
print(sd)

vr=np.var(df1)
print(vr)

print(df1)


df1.plot.scatter(x='GarageArea', y='SalePrice', edgecolors='r')

z = np.abs(stats.zscore(df1))#z-score is the number of standard deviations from the mean value

print(z)

threshold = 3

print(np.where(z > 3))

print(z[1373][0]) #which mean z[1373][0] have a Z-score higher than 3.

df_o = df1[(z < 3).all(axis=1)]

df_o.plot.scatter(x='GarageArea', y='SalePrice', edgecolors='r');

plt.show()
