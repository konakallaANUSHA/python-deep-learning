from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read in the file with pandas
diabetes = pd.read_csv('diabetes.csv')

# Drop any nan values if any exist
diabetes_data = diabetes.select_dtypes(include=[np.number]).interpolate().dropna()

# If the column is non-numeric, dummify the data
for column in diabetes_data:
    if np.issubdtype(diabetes_data[column].dtype, np.number) == False:
        diabetes_data = pd.get_dummies(
            diabetes_data,
            columns=[column]
        )

corr = diabetes_data.corr()
print(corr)
print (corr['Outcome'].sort_values(ascending=False), '\n')


# Determine data(X) vs target(y)
X = diabetes_data.drop(['Outcome'], axis=1)
y = diabetes_data['Outcome']

# Split the data set into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# LinearSVC accuracy
c = LinearSVC().fit(X_train, y_train)
classifer = LinearSVC()

#Training Model
classifer.fit(X_train,y_train)
#predicting Output
y_predict = classifer.predict(X_test)
accuracy = accuracy_score(y_test,y_predict)
print("SVM accuracy:", accuracy)

# Gaussian Bayes accuracy
g = GaussianNB().fit(X_train, y_train)
classifer = GaussianNB()
#Training the Model
classifer.fit(X_train,y_train)
#predicting the Output
y_predict = classifer.predict(X_test)
accuracy = accuracy_score(y_test,y_predict)
print("naive accuracy:", accuracy)

# KNN accuracy
classifer = KNeighborsClassifier(n_neighbors=3)
#Training the Model
classifer.fit(X_train,y_train)
#predicting the Output
y_predict = classifer.predict(X_test)

accuracy = accuracy_score(y_test,y_predict)
print("KNN accuracy :" , accuracy)