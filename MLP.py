# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 15:56:16 2018

@author: Gayathri Venkatesh
"""

import os, sys

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Embedding
from keras.layers import Dense, LSTM, Dropout, Embedding
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Sequential

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.utils import np_utils
import re

import nltk
from nltk.corpus import stopwords
from nltk.corpus import RegexpTokenizer as regextoken
#LOADING THE DATA
os.getcwd()
os.chdir('D:/INSOFE Internship/CUTE4/train')

data = pd.read_csv("train.csv",header=0)
test = pd.read_csv("test.csv",header=0)

#UNDERSTAND THE DATA
data.shape 

test.shape 

data.columns

test.columns

#DISPLAY THE INDEX
data.index

test.index

#Understand the distribution of the data
data.describe(include='all')

#DISPLAY THE DATA TYPE OF EACH VARIABLE
data.dtypes

#Check the unique classes in categories column
np.unique(data['categories'])

pd.value_counts(data['categories'])

np.size(np.unique(data['categories']))

#Check the data for missing values
data.isnull().sum()

#Drop the columns which has na values
data=data.dropna(how='any')

#Check whether the missing values are removed
data.isnull().sum()

#Categories column is of 'object' type,changing it to category data type
data['categories'] = data['categories'].astype('category')
#test['categories'] = test['categories'].astype('category')

#Drop the ID column from both train and test data
data.drop('ID', axis=1, inplace=True)
test.drop('ID', axis=1, inplace=True)

#Check the data type after changing the data type to categorical
data.dtypes

#Check the columns 
data.columns

#Label Encoder encodes the categorical classes as different numbers
#Since the 'categories' column is the column to be predicted in test,set labels
#to the differone hot encoded verson
LabelEncoder = preprocessing.LabelEncoder()
labels = LabelEncoder.fit_transform(data['categories'])
y_labels= np_utils.to_categorical(labels, 6)
set(labels)
labels.shape

#Split the data into train and validation
X_train, X_validation, y_train, y_validation = train_test_split(data.iloc[:,1], y_labels, test_size=0.3, random_state=123) 

#Converting the X_train and X_validation is now converted to array from 
#dataframe, for ease of use convert them to list so that they can be used in 
#tokenizer

x_train_sent=[i for i in X_train]

x_validation_sent=[j for j in X_validation ]

test_sent=[k for k in test.iloc[:,0]]



len(x_train_sent)


for i in range(0,32057):
    x_train_sent[i]=re.sub("[^a-zA-Z]", " ",str(x_train_sent[i]))
    x_train_sent[i].lower()

for i in range(0,13739):
    x_validation_sent[i]=re.sub("[^a-zA-Z]", " ",str(x_validation_sent[i]))
    x_validation_sent[i].lower()

for i in range(0,11455):
    test_sent[i]=re.sub("[^a-zA-Z]", " ",str(test_sent[i]))
    test_sent[i].lower()
    

complete=x_train_sent+x_validation_sent+test_sent

# Prepare tokenizer for train and validation
tokenizer1 = Tokenizer()
tokenizer1.fit_on_texts(complete)

vocab_Size = len(tokenizer1.word_index) + 1
print('Found %s unique tokens.' % vocab_Size)

# integer encode the documents
#train
sequences_train = tokenizer1.texts_to_sequences(x_train_sent)

# integer encode the documents
#validation
sequences_validation = tokenizer1.texts_to_sequences(x_validation_sent)

# integer encode the documents
#test
sequences_test = tokenizer1.texts_to_sequences(test_sent)

MAX_SEQUENCE_LENGTH = 700

train_pad = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH,padding='post')

print('Shape of train_data tensor:', train_pad.shape)

validation_pad = pad_sequences(sequences_validation, maxlen=MAX_SEQUENCE_LENGTH,padding='post')

print('Shape of validation_data tensor:', validation_pad.shape)

test_pad = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH,padding='post')

print('Shape of test_data tensor:', test_pad.shape)

###MLP model
mlp_model3=Sequential()
mlp_model3.add(Embedding(vocab_Size, 2, input_length=MAX_SEQUENCE_LENGTH ))
mlp_model3.add(Flatten())
mlp_model3.add(Dense(500, input_shape=(MAX_SEQUENCE_LENGTH,), activation='relu',kernel_initializer = "glorot_normal"))
mlp_model3.add(Dense(100, activation='relu',kernel_initializer = "glorot_normal"))
mlp_model3.add(Dense(6, activation='softmax',kernel_initializer = "normal"))

mlp_model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(mlp_model3.summary())

#Fit the model
mlp_model3.fit(train_pad, y_train,validation_data=(validation_pad, y_validation), epochs=3,batch_size=64,verbose=2)

# evaluate the model
loss, accuracy = mlp_model3.evaluate(train_pad, y_train, verbose=0)
print('Accuracy: %f' % (accuracy*100))

#Fit the model
#cnn_model.fit(train_pad, y_train,validation_data=(validation_pad, y_validation), epochs=12, batch_size=64,verbose=2)

validation_pred = mlp_model3.predict(validation_pad)

# evaluate the model
loss, accuracy = mlp_model3.evaluate(validation_pad, y_validation, verbose=0)
print('Accuracy: %f' % (accuracy*100))

test_pred = mlp_model3.predict(test_pad)

#test_pred

test_predictions =[]
for i in test_pred:
    test_predictions.append(np.argmax(i))


test_predictions = LabelEncoder.inverse_transform(test_predictions)

len(test_predictions)

#test_predictions
test = pd.read_csv("test.csv",header=0)

copy_test=test


output = {'ID':copy_test['ID'],'categories': test_predictions}
output_df = pd.DataFrame(data=output)

pd.value_counts(output['categories'])

output_df.to_csv('predict4.csv',index=False)



###MLP model
mlp_model4=Sequential()
mlp_model4.add(Embedding(vocab_Size, 2, input_length=MAX_SEQUENCE_LENGTH ))
mlp_model4.add(Flatten())
mlp_model4.add(Dense(500, input_shape=(MAX_SEQUENCE_LENGTH,), activation='relu',kernel_initializer = "glorot_normal"))
mlp_model4.add(Dropout(0.2))
mlp_model4.add(Dense(100, activation='relu',kernel_initializer = "glorot_normal"))
mlp_model4.add(Dropout(0.3))
mlp_model4.add(Dense(6, activation='softmax',kernel_initializer = "normal"))

mlp_model4.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(mlp_model4.summary())

#Fit the model
mlp_model4.fit(train_pad, y_train,validation_data=(validation_pad, y_validation), epochs=3,batch_size=64,verbose=2)

# evaluate the model
loss, accuracy = mlp_model4.evaluate(train_pad, y_train, verbose=0)
print('Accuracy: %f' % (accuracy*100))

#Fit the model
#cnn_model.fit(train_pad, y_train,validation_data=(validation_pad, y_validation), epochs=12, batch_size=64,verbose=2)

validation_pred = mlp_model4.predict(validation_pad)

# evaluate the model
loss, accuracy = mlp_model4.evaluate(validation_pad, y_validation, verbose=0)
print('Accuracy: %f' % (accuracy*100))

test_pred = mlp_model4.predict(test_pad)

#test_pred

test_predictions =[]
for i in test_pred:
    test_predictions.append(np.argmax(i))


test_predictions = LabelEncoder.inverse_transform(test_predictions)

len(test_predictions)

#test_predictions
test = pd.read_csv("test.csv",header=0)

copy_test=test


output = {'ID':copy_test['ID'],'categories': test_predictions}
output_df = pd.DataFrame(data=output)

pd.value_counts(output['categories'])

output_df.to_csv('predict5.csv',index=False)