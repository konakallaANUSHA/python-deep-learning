import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

wine = pd.read_csv('/Users/anushakonakalla/Downloads/winequality-red.csv')

wine.info()

##Null values
nulls = pd.DataFrame(wine.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)


#correlation
corr = wine.corr()
print(corr)
print (corr['quality'].sort_values(ascending=False)[:4], '\n')

#correlation plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(wine.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(wine.columns)
ax.set_yticklabels(wine.columns)
plt.show()


train_df, test_df = train_test_split(wine, test_size=0.3, random_state=0)

X_train = train_df.drop(['quality', 'fixed acidity', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density' ,'pH', 'volatile acidity'], axis=1)
Y_train = np.log(train_df["quality"])
X_test = test_df.drop(['quality', 'fixed acidity', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density' ,'pH', 'volatile acidity'], axis=1)
Y_test = np.log(test_df["quality"])


lr = linear_model.LinearRegression()

model = lr.fit(X_train, Y_train)

print('Coefficients: \n', lr.coef_)
print('intercept: \n', lr.intercept_)



##Evaluate the performance

predictions = model.predict(X_test)

RSQR = model.score(X_test, Y_test)
print("R^2 is:", RSQR)

print("Variance score: %.2f" % r2_score(Y_test,predictions))

MSE = mean_squared_error(Y_test, predictions)#Mean square error
print('MSE is:', MSE)

RMSE = np.sqrt(MSE)#root mean square error
print("RMSE:", RMSE)

MAE = mean_absolute_error(Y_test, predictions) #Mean absolute error
print('MAE is:', MAE)

##visualize

actual_values = Y_test

plt.scatter(predictions, actual_values,  color='y')

plt.xlabel('Predicted values')

plt.ylabel('Actual values')

plt.title('Multiple Regression Model')

plt.show()