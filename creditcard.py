#!/usr/bin/env python
# coding: utf-8

#importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,accuracy_score
import keras
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sns
import matplotlib.pyplot as plt

#reading the data
data = pd.read_csv("creditcard.csv") 
data.head()

#exploring the data
data.shape
data.describe()

#checking if missing data is present
data.isnull()

#features correlation
plt.figure(figsize=(14,14))
correlate=data.corr()
sns.heatmap(correlate,xticklabels=correlate.columns,yticklabels=correlate.columns,linewidths=0.1)
plt.show()

#identifying predictor features and target variables and separating them
feature = [col for col in data.columns if col not in ['Class','Time']]
X=data[feature]
Y=data.Class
#splitting the data into train and test datasets
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)

# # Logistic Regression

#applying logistic regression and training the model
logregmodel = LogisticRegression()
logregmodel.fit(X_train,Y_train)

#testing the model
predict=logregmodel.predict(X_test)
print(classification_report(Y_test,predict))
print("Accuracy:",accuracy_score(Y_test,predict))

# # Artificial Neural Network

#applying artificial neural network model
classify=Sequential()
classify.add(Dense(units=15,kernel_initializer='uniform',activation='relu',input_dim=29))
classify.add(Dense(units=15,kernel_initializer='uniform',activation='relu'))
classify.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
classify.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#fitting the ann
classify.fit(X_train,Y_train,batch_size=16,epochs=200)

#testing the model
predictions=classify.predict(X_test)
predictions=(predictions>0.5)
print(classification_report(Y_test,predictions))
print("Accuracy:",accuracy_score(Y_test,predictions))
