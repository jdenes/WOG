#!/usr/bin/env python
# coding: utf-8

# ### Demo: using the RNE (http://bit.ly/data-RNE) to explore municipal councillors' representativeness

# In[1]:


import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


data = pd.read_csv("1-rne-cm.txt", sep='\t', encoding='ISO-8859-1', low_memory=False)


# In[3]:


data.shape


# In[4]:


data.head()


# In[5]:


data['Libellé de département (Maires)'].value_counts().plot(kind='bar', figsize=(20,5))


# In[6]:


data['Code sexe'].value_counts(normalize=True)


# In[7]:


data['Libellé de la profession'].value_counts().head(30).plot(kind='barh', figsize=(20,8))


# In[8]:


data["Nationalité de l'élu"].value_counts(normalize=True)


# In[9]:


data['Date de naissance'].describe()


# In[10]:


data['Date de naissance clean'] = data['Date de naissance'].astype("datetime64")


# In[11]:


data['Date de naissance clean'].describe()


# In[24]:


data['age'] = (pd.Timestamp('now') - data['Date de naissance clean']).astype('<m8[Y]')


# In[25]:


data['age'].describe()


# In[12]:


data.groupby(data['Date de naissance clean'].dt.year).count()['Code sexe'].plot(figsize=(15, 5))


# In[13]:


data.groupby(['Libellé de département (Maires)', 'Code sexe']).size().unstack().plot.bar(figsize=(25, 5))


# ### Let's try data science!

# In[14]:


data = data[pd.notnull(data["Code sexe"]) & pd.notnull(data["Date de naissance clean"])
            & pd.notnull(data["Code profession"])]


# In[15]:


data['Date de naissance number'] = data['Date de naissance clean'].dt.strftime("%Y%m%d").astype(int)


# In[16]:


data['Date de naissance number'].describe()


# In[17]:


data['Gender'] = data['Code sexe'].apply(lambda x: 0 if x=='F' else 1)


# In[18]:


data['Gender'].describe()


# In[19]:


train, test = train_test_split(data, test_size=0.25)
X_train = train[['Code profession', 'Date de naissance number']]
X_test = test[['Code profession', 'Date de naissance number']]
Y_train = train["Gender"]
Y_test = test["Gender"]


# In[20]:


model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, Y_train)


# In[21]:


Y_pred = model.predict(X_test)
print(Y_pred)


# In[22]:


accuracy_score(Y_test, Y_pred)

