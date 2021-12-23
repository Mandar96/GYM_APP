#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# In[53]:


data = pd.read_excel(r'D:\Data_Science\DS_Files\dataGYM.xlsx')


# In[54]:


data.head()


# In[55]:


df = data.copy()
del df['BMI']
del df['Class']
label_encoder = LabelEncoder()
df['Prediction'] = label_encoder.fit_transform(df['Prediction'])


# In[56]:


X = df.iloc[:,:-1]
y = df.iloc[:,-1]


# In[57]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

model_GYM = RandomForestClassifier(n_estimators=20)

model_GYM.fit(X_train, y_train)

print(model_GYM)


# In[59]:


# make predictions
expected = y_test
predicted = model_GYM.predict(X_test)


# In[61]:


# summarize the fit of the model
#Correction
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


# In[62]:


import pickle

pickle.dump(model_GYM, open("Model_GYM.pkl", "wb"))

model = pickle.load(open("Model_GYM.pkl", "rb"))

print(model.predict([[40,5.6,70]]))


# In[ ]:





# In[ ]:





# In[ ]:




