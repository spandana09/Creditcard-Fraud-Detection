#!/usr/bin/env python
# coding: utf-8

# In[6]:


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


# In[2]:


#reading the data
data = pd.read_csv("creditcard.csv") 
data.head()


# In[3]:


#exploring the data
data.shape
data.describe()


# In[4]:


#checking if missing data is present
data.isnull()


# In[8]:


#features correlation
plt.figure(figsize=(14,14))
correlate=data.corr()
sns.heatmap(correlate,xticklabels=correlate.columns,yticklabels=correlate.columns,linewidths=0.1)
plt.show()


# # Logistic Regression

# In[ ]:


#identifying predictor features and target variables and separating them
feature = [col for col in data.columns if col not in ['Class','Time']]
X=data[feature]
Y=data.Class
#spliiting the data into train and test datasets
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)


# In[ ]:


#applying logistic regression and training the model
logregmodel = LogisticRegression()
logregmodel.fit(X_train,Y_train)


# In[ ]:


#testing the model
predict=logregmodel.predict(X_test)


# In[ ]:


print(classification_report(Y_test,predict))
print("Accuracy:",accuracy_score(Y_test,predict))


# # Artificial Neural Network

# In[ ]:


#applying artificial neural network model
classify=Sequential()
classify.add(Dense(units=15,kernel_initializer='uniform',activation='relu',input_dim=29))
classify.add(Dense(units=15,kernel_initializer='uniform',activation='relu'))
classify.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
classify.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#fitting the ann
classify.fit(X_train,Y_train,batch_size=16,epochs=200)


# In[ ]:


#testing the model
predictions=classify.predict(X_test)
predictions=(predictions>0.5)
print(classification_report(Y_test,predictions))
print("Accuracy:",accuracy_score(Y_test,predictions))

