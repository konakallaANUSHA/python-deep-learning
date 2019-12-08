import sys

from keras.models import Sequential
from keras.layers.core import Dense
import pandas as pd
import numpy as np
from keras.optimizers import SGD
from tensorboardcolab import *
tbc=TensorBoardColab()

np.set_printoptions(threshold=sys.maxsize)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("heart.csv")


X= dataset.iloc[:, [0,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
y = dataset['target']

sc = StandardScaler()
X_scaled_array = sc.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled_array, columns = X.columns)
print(X_scaled)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled,y,test_size=0.25, random_state=0)
print(X_train.shape[1])
print(X_train)
print(X_test)

model = Sequential() # create model
model.add(Dense(50, input_dim=X_train.shape[1], activation='relu')) # hidden layer
model.add(Dense(20, activation='relu'))
#model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # output layer

epochs = 100
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X_train,Y_train, validation_data=(X_test, Y_test), epochs=epochs, batch_size=64,callbacks=[TensorBoardColabCallback(tbc)])
# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


preds = model.predict(X_test)
print(preds)