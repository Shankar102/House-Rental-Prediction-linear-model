#!/usr/bin/env python
# coding: utf-8

# Let's create a ML model to predict rent of house.
# Here is the data - Link
# Make sure you do data wrangling & get useful insights/visualizations
# Create models using Linear Regressions or variations of it
# Think harder about data preprocessing
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
import seaborn as sns
print("Done")


# In[2]:


df=pd.read_csv('data/House_Rental_Dataset.csv')
df.head(5)


# In[3]:


df.isnull().sum()


# In[4]:


df.describe()


# In[7]:


df.columns


# In[9]:


df.info


# In[21]:


a=df['Price'].unique()
a


# In[16]:


df['Price'].nunique()


# In[39]:


plt.figure(figsize=(15,6))
sns.countplot('Price',data=df.head(100))
plt.xticks(rotation=90)
plt.show()


# In[31]:


df['Price'].value_counts()


# In[32]:


df['Sqft'].value_counts()


# In[36]:


plt.figure(figsize=(15,6))
sns.countplot('Sqft',data=df.head(100))
plt.xticks(rotation=90)
plt.show()


# In[35]:


sns.lmplot(x="Price",y="Sqft",data=df,order=2,ci=None)


# In[43]:


X=np.array(df['Price']).reshape(-1,1)
y=np.array(df['Sqft']).reshape(-1,1)

df.dropna(inplace=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

lg=LinearRegression() 
lg.fit(X_train,y_train)
print(lg.score(X_test,y_test))


# In[44]:


df.fillna(method='ffill',inplace=True)


# In[46]:


y_pred=lg.predict(X_test)
plt.scatter(X_test,y_test,color='b')
plt.plot(X_test,y_pred,color='k')
plt.show()


# In[47]:


from sklearn.metrics import mean_absolute_error,mean_squared_error
mae=mean_absolute_error(y_true=y_test,y_pred=y_pred)
mse=mean_squared_error(y_true=y_test,y_pred=y_pred)
rmse=mean_squared_error(y_true=y_test,y_pred=y_pred,squared=False)
print("mae",mae)
print("mse",mse)
print("rmse",rmse)


# In[89]:


df_binary500=df[:][:500]

sns.lmplot(x="Price",y="Sqft",data=df_binary500,order=700,ci=None)


# In[99]:


X=np.array(df['Price']).reshape(-1,1)
y=np.array(df['Sqft']).reshape(-1,1)

df.dropna(inplace=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

lg=LinearRegression() 
lg.fit(X_train,y_train)
print(lg.score(X_test,y_test))


# In[101]:


y_pred=lg.predict(X_test)
plt.scatter(X_test,y_test,color='b')
plt.plot(X_test,y_pred,color='k')
plt.show()


# In[98]:


df.shape


# In[102]:


from sklearn.metrics import mean_absolute_error,mean_squared_error
mae=mean_absolute_error(y_true=y_test,y_pred=y_pred)
mse=mean_squared_error(y_true=y_test,y_pred=y_pred)
rmse=mean_squared_error(y_true=y_test,y_pred=y_pred,squared=False)
print("mae",mae)
print("mse",mse)
print("rmse",rmse)


# In[ ]:




