from keras.models import Sequential
from keras.layers.core import Dense
import pandas as pd
from sklearn.model_selection import train_test_split


dataset = pd.read_csv("Breas Cancer.csv")
# print(dataset)

X = dataset.iloc[:, 2:32].values
dataset['diagnosis'].replace('M', 1,inplace=True)
dataset['diagnosis'].replace('B', 0,inplace=True)
y = dataset.iloc[:, 1].values
print(y)

X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.25, random_state=0)



my_first_nn = Sequential() # create model
my_first_nn.add(Dense(20, input_dim=30, activation='relu')) # hidden layer
my_first_nn.add(Dense(20, activation='relu'))
my_first_nn.add(Dense(20, activation='relu'))

my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100, verbose=0,
                                     initial_epoch=0)
print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, Y_test, verbose=0))
