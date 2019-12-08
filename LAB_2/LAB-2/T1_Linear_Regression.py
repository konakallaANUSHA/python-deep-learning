import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.optimizers import Adam
from tensorboardcolab import *
tbc=TensorBoardColab()

data = pd.read_csv('Breast_cancer_data.csv')
data.head(10)
print(data.info())

X = data.iloc[:,0:5].values
Y = data.iloc[:,5].values
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(3, kernel_initializer="uniform", activation = 'relu', input_dim = 5))
model.add(Dense(3, kernel_initializer="uniform", activation = 'relu'))
model.add(Dense(1, kernel_initializer="uniform", activation = 'sigmoid'))
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs = 50,callbacks=[TensorBoardColabCallback(tbc)])
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Prediction
y_pred = model.predict(X_test)
print(y_pred)

from sklearn.metrics import confusion_matrix

#cm = confusion_matrix(y_test, y_pred)
#print(cm)