#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split


# In[11]:


df=pd.read_csv('medical_insurance.csv', header=0)
df.head()


# In[12]:


df.info()


# In[13]:


df.describe(include='all')


# In[14]:


df.isna().value_counts()


# In[15]:


dummy_vars=pd.get_dummies(df[['sex','smoker','region']])
dummy_vars


# In[16]:


#rename the columns
df.drop('smoker',axis=1,inplace=True)

dummy_vars=dummy_vars.rename(columns={'sex_female':'female','sex_male':'male','smoker_yes':'smoker','region_northeast':'northeast','region_northwest':'northwest','region_southeast':'southeast','region_southwest':'southwest'})
dummy_vars.drop('smoker_no', axis=1,inplace=True)
dummy_vars


# In[17]:


df=pd.concat([df,dummy_vars],axis=1)
df.drop(['sex','region'],axis=1,inplace=True)
df.head()


# In[18]:


sns.regplot(x='bmi',y='charges',data=df)


# In[19]:


sns.boxplot(data=df,x='smoker',y='charges')


# In[20]:


df.corr()


# In[21]:


lm= LinearRegression()
lm


# In[22]:


x_data=df[['smoker']]
y_data=df['charges']
lm.fit(x_data,y_data)

print(f'The R^2 score of the given model is {lm.score(x_data,y_data)}.')


# In[23]:


x_all=df.drop(['charges'],axis=1)
lm.fit(x_all,y_data)
r2_score1=lm.score(x_all,y_data)
print(f'The R^2 score of the given model, with all attributes except "smoker" is {r2_score}.')


# In[24]:


Input=[('scale',StandardScaler()),('polynomial',PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
pipe=Pipeline(Input)
x_all=x_all.astype(float)
pipe.fit(x_all,y_data)
yhat=pipe.predict(x_all)
r2_score2=r2_score(y_data,yhat)
print(f'The R^2 score of the model after the training pipeline is {r2_score2}.')


# In[25]:


print(f'Here the model performance improved by {r2_score2-r2_score1} after training pipeline.')


# In[26]:


x_train,x_test,y_train,y_test=train_test_split(x_all,y_data,test_size=.2,random_state=1)


# In[27]:


RidgeModel=Ridge(alpha=0.1)
RidgeModel.fit(x_train,y_train)
yhat=RidgeModel.predict(x_test)
r2_score3=r2_score(y_test,yhat)
print(f'The R^2 score of the model after Ridge Regression is {r2_score3}.')


# In[28]:


print(f'Here the model performance degraded by {r2_score3-r2_score2} after training Ridge Regression at alpha=0.1.')


# In[29]:


pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train)
x_test_pr=pr.fit_transform(x_test)
RidgeModel.fit(x_train_pr,y_train)
y_hat=RidgeModel.predict(x_test_pr)
r2_score4=r2_score(y_test,y_hat)
print(f'The R^2 score of the model after polynomial transformation on Ridge Regression is {r2_score4}.')


# In[30]:


print(f'Here the model performance improved by {r2_score4-r2_score3} after training Ridge Regression at alpha=0.1.')


# In[32]:


import pickle

# Assuming you have already trained your model (RidgeModel in this case)
# Save the model to disk
filename = 'insurance_model.pkl'
pickle.dump(RidgeModel, open(filename, 'wb'))


# In[33]:


pipeline_filename = 'insurance_pipeline.pkl'
pickle.dump(pipe, open(pipeline_filename, 'wb'))
