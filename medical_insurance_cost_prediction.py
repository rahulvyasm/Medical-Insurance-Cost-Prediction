# In[1]:

    
import pandas as pd
import seaborn as sns
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge

# In[2]:

    
df=pd.read_csv('medical_insurance.csv', header=0)
print(f"{df.heaf()}")

# In[3]:

    
print(f"\n{df.info()}")

# In[4]:

    
print(f"\n{df.describe(include='all')}")

# In[5]:

    
print(f"\n{df.isna().value_counts()}")

# In[6]:

    
dummy_vars=pd.get_dummies(df[['sex','smoker','region']])
print(f"\n{dummy_vars}")

# In[7]:

    
#renameing the columns
df.drop('smoker',axis=1,inplace=True)

# In[9]:

    
dummy_vars=dummy_vars.rename(columns={'sex_female':'female','sex_male':'male','smoker_yes':'smoker','region_northeast':'northeast','region_northwest':'northwest','region_southeast':'southeast','region_southwest':'southwest'})
dummy_vars.drop('smoker_no', axis=1,inplace=True)
print(f"\n{dummy_vars}")

# In[10]:

    
df=pd.concat([df,dummy_vars],axis=1)
df.drop(['sex','region'],axis=1,inplace=True)
print(f"\n{df.head()}")

# In[11]:

    
#using a Regression Plot
print(f"\n{sns.regplot(x='bmi',y='charges',data=df)}")

# In[12]:

    
#using a Box Plot
print(f"{sns.boxplot(data=df,x='smoker',y='charges')}")


# In[13]:


df.corr()


# In[14]:


lm= LinearRegression()
print(f"{lm}")


# In[15]:


x_data=df[['smoker']]
y_data=df['charges']
lm.fit(x_data,y_data)

print(f'The R^2 score of the given model is {lm.score(x_data,y_data)}.')


# In[16]:


x_all=df.drop(['charges'],axis=1)
lm.fit(x_all,y_data)
r2_score1=lm.score(x_all,y_data)
print(f'The R^2 score of the given model, with all attributes except "smoker" is {r2_score}.')


# In[17]:


Input=[('scale',StandardScaler()),('polynomial',PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
pipe=Pipeline(Input)
x_all=x_all.astype(float)
pipe.fit(x_all,y_data)
yhat=pipe.predict(x_all)
r2_score2=r2_score(y_data,yhat)
print(f'The R^2 score of the model after the training pipeline is {r2_score2}.')


# In[18]:


print(f'Here the model performance improved by {r2_score2-r2_score1} after training pipeline.')


# In[19]:


x_train,x_test,y_train,y_test=train_test_split(x_all,y_data,test_size=.2,random_state=1)


# In[20]:


RidgeModel=Ridge(alpha=0.1)
RidgeModel.fit(x_train,y_train)
yhat=RidgeModel.predict(x_test)
r2_score3=r2_score(y_test,yhat)
print(f'The R^2 score of the model after Ridge Regression is {r2_score3}.')


# In[21]:


print(f'Here the model performance degraded by {r2_score3-r2_score2} after training Ridge Regression at alpha=0.1.')


# In[22]:


pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train)
x_test_pr=pr.fit_transform(x_test)
RidgeModel.fit(x_train_pr,y_train)
y_hat=RidgeModel.predict(x_test_pr)
r2_score4=r2_score(y_test,y_hat)
print(f'The R^2 score of the model after polynomial transformation on Ridge Regression is {r2_score4}.')


# In[23]:


print(f'Here the model performance improved by {r2_score4-r2_score3} after training Ridge Regression at alpha=0.1.')


# In[24]:


# Save the model to disk
filename = 'insurance_model.pkl'
pickle.dump(RidgeModel, open(filename, 'wb'))


# In[25]:


pipeline_filename = 'insurance_pipeline.pkl'
pickle.dump(pipe, open(pipeline_filename, 'wb'))
