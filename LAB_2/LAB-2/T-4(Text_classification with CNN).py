import re
from keras.layers.convolutional import MaxPooling1D
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Conv1D, GlobalMaxPooling1D, Embedding
import matplotlib
import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
from sklearn import metrics
import seaborn as sns

import matplotlib.pyplot as plt
#git hub https://github.com/Stass88/lattelecom
df_train = pd.read_csv('sentiment-analysis-on-movie-reviews/train.tsv', sep='\t')
df_test = pd.read_csv('sentiment-analysis-on-movie-reviews/test.tsv', sep='\t')
##Descriptive analyse
#this should help you to decide whether to use STOP WORDS or not.
#This part of code is just great analytical tool


##Data preprocessing
#we make text lower case and leave only letters from a-z and digits
df_train['Phrase'] = df_train['Phrase'].str.lower()
df_train['Phrase'] = df_train['Phrase'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
df_test['Phrase'] = df_test['Phrase'].str.lower()
df_test['Phrase'] = df_test['Phrase'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
X_train = df_train.Phrase
y_train = df_train.Sentiment
max_fatures = 2000
tokenize = Tokenizer(num_words=max_fatures, split=' ')
tokenize.fit_on_texts(X_train.values)
X_test = df_test.Phrase
X_train = tokenize.texts_to_sequences(X_train)
X_test = tokenize.texts_to_sequences(X_test)
max_lenght = max([len(s.split()) for s in df_train['Phrase']])
X_train = pad_sequences(X_train, max_lenght)
X_test = pad_sequences(X_test, max_lenght)
print(X_train.shape)
print(X_test.shape)

##Model building
model=Sequential()
model.add(Embedding(max_fatures, output_dim=100,input_length=48))
model.add(Conv1D(filters=64, kernel_size=5, activation='relu', padding='causal'))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=5, activation='relu', padding='causal'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.7))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(5,activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train, epochs=40, verbose=True,  batch_size=1024)

