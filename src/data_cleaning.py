#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split


# In[2]:


# Load data
data = np.load('../data/dataset.npz', allow_pickle=True)

X_train = data['X_train'] 
y_train = data['y_train'] 
X_test = data['X_test']
y_test = data['y_test']
X_val = data['X_val']
y_val = data['y_val']

# Concatenate numpy arrays 
X = np.concatenate([X_train, X_test, X_val]) 
y = np.concatenate([y_train, y_test, y_val]) 

# Create dataframe 
df = pd.DataFrame(
    {'document': X,
     'summary': y
    })

# Make sure there are no missing values
df.dropna(inplace=True)


# In[3]:


# Convert all text to lower case
df['document'] = df['document'].str.lower()
df['summary'] = df['summary'].str.lower()


# In[4]:


# Remove special characters
document = df['document'].tolist()
summary = df['summary'].tolist()

document = [re.sub(r'\r\n|\r|\n', ' ', _) for _ in document] # replace new line tags w/ space
document = [re.sub(r"[-]", ' ', _) for _ in document] # replace dashes w/ space
document = [re.sub('[^a-zA-Z0-9 ]+', '', _) for _ in document] # remove any other special characters
document = [re.sub(' +', ' ', _) for _ in document] # replace multiple spaces w/ single

summary = [re.sub(r'\r\n|\r|\n', ' ', _) for _ in summary]
summary = [re.sub(r"[-]", ' ', _) for _ in summary]
summary = [re.sub('[^a-zA-Z0-9 ]+', '', _) for _ in summary]
summary = [re.sub(' +', ' ', _) for _ in summary]

# Update changes to df
df['document'] = document
df['summary'] = summary


# In[6]:


# Split data into train, test, and val sets
X_train, X_val, y_train, y_val = train_test_split(df["document"], df["summary"], train_size=0.8, random_state=42)

# Format as numpy arrays
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)
X_val = np.asarray(X_val)
y_val = np.asarray(y_val)


# In[8]:


# Save the cleaned data
np.savez('../data/dataset_cleaned.npz' , X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, X_val=X_val, y_val=y_val)


# In[ ]:




