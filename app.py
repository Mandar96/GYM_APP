#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install gunicorn')


# In[2]:


get_ipython().system('pip install flask')


# In[3]:


get_ipython().system('pip install itsdangerous')


# In[4]:


get_ipython().system('pip install Jinja2')


# In[5]:


get_ipython().system('pip install MarkupSafe')


# In[1]:


import flask
from flask import Flask, request , jsonify, render_template
#import jinja2
import numpy as np
import pandas as pd
import pickle


# In[2]:


from flask import Flask, request,jsonify,render_template
import pickle
import pandas as pd
import numpy as np


# In[3]:


app = Flask(__name__)
filename = 'Model_GYM.pkl'
model = pickle.load(open(filename, 'rb'))


# In[4]:


@app.route('/')
def man():
    return render_template('home.html')


# In[5]:


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    arr = np.array([[data1, data2, data3]])
    pred = model.predict(arr)
    return render_template('after.html',data=pred)


# In[ ]:


if __name__ == "__main__":
    app.run()

