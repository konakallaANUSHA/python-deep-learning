import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import  train_test_split
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
# Create SVM classification object
model = GaussianNB()
# there is other distribution for multinomial classes like Bernoulli Naive Bayes, Refer link
# Train the model using the training sets and check score
X_train = pd.read_csv('/Users/anushakonakalla/Downloads/glass.csv')
train_df, test_df = train_test_split(X_train, test_size=0.3, random_state=0)



X_train = train_df.drop("Type",axis=1)

Y_train = train_df["Type"]
X_test = test_df.drop("Type",axis=1)
Y_test = test_df["Type"]
combine = [train_df, test_df]



model.fit(X_train, Y_train)

predict_train = model.predict(X_train)
print('Target on train data',predict_train)

# Accuray Score on train dataset
accuracy_train = accuracy_score(Y_train,predict_train)
print('accuracy_score on train dataset : ', accuracy_train)

# predict the target on the test dataset
predict_test = model.predict(X_test)
print('Target on test data',predict_test)

# Accuracy Score on test dataset
accuracy_test = accuracy_score(Y_test,predict_test)
print('accuracy_score on test dataset : ', accuracy_test)

report=classification_report( Y_train,predict_train)
print(report)

report1=classification_report( Y_test, predict_test)
print(report1)
