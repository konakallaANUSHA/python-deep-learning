import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

df = pd.read_csv('/Users/anushakonakalla/lab/kc_house_data.csv')

df.info()

##Null values
nulls = pd.DataFrame(df.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)

df['year']=df.date.str[0:4]
df['month']=df.date.str[4:6]
df['day']=df.date.str[6:8]
df['year'] = [int(i) for i in df['year']]
print(df['year'])
df['month'] = [int(i) for i in df['month']]
print(df['month'])
df['day'] = [int(i) for i in df['day']]
print((df['day']))
df['days'] = (df['year']-2012)*365 + (df['month']-1)*30 + (df['day']-1)
print(df['days'])
df = df.drop(['date','year','month','day','zipcode'],1)

#correlation
corr = df.corr()
print(corr)
print (corr['price'].sort_values(ascending=False), '\n')

#correlation plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='PiYG', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(df.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(df.columns)
ax.set_yticklabels(df.columns)
plt.show()



df.info()

train_df, test_df = train_test_split(df, test_size=0.3, random_state=0)




X_train = train_df.drop(['price','waterfront','days','floors','yr_renovated', 'sqft_lot', 'sqft_lot15', 'yr_built','condition','long', 'id'], axis=1)
Y_train = np.log(train_df["price"])
X_test = test_df.drop(['price', 'waterfront','days', 'floors','yr_renovated', 'sqft_lot', 'sqft_lot15', 'yr_built','condition','long', 'id'], axis=1)
Y_test = np.log(test_df["price"])


lr = linear_model.LinearRegression()

model = lr.fit(X_train, Y_train)

print('Coefficients: \n', lr.coef_)
print('intercept: \n', lr.intercept_)



##Evaluate the performance

predictions = model.predict(X_test)

RSQR = model.score(X_test, Y_test)
print("R^2 is:", RSQR)

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