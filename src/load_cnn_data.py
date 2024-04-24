import datasets 
import pandas as pd
from datasets import load_dataset
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

small_dataset = False

dataset = load_dataset("cnn_dailymail", "3.0.0") 

if small_dataset:
    max_len = 100 # Small Dataset
else:
    max_len = 1000 # Large Dataset

# retrieve up to the specified number of rows 
# Data is originially in a DatasetDict object with separate Datasets for train, test, and validation. We will be using a subset of the train dataset. 
dataset = dataset['train'][:max_len]

# final data structure 
data = []

# Data is originally in an Arrow DatasetObject. By looping over all three arrays in Dataset object and zipping, we can extract each article, highlight, and id and place into a dictionary in a list. 
for article, highlight in zip(dataset['article'], dataset['highlights']):
    data.append({'article' : article, 'highlights' : highlight})

data = pd.DataFrame.from_dict(data)

data.head()

train, test = train_test_split(data, test_size=.2)
train, val = train_test_split(train, test_size=.25)

X_train = train.iloc[:,0]
y_train = train.iloc[:,1]
X_test = test.iloc[:,0]
y_test = test.iloc[:,1]
X_val = val.iloc[:,0]
y_val = val.iloc[:,1]

print("Train: " + str(len(X_train)))
print("Test: " + str(len(X_test)))
print("Val: " + str(len(X_val)))

if small_dataset:
    np.savez('../data/cnn_dailymail_small.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, X_val=X_val, y_val=y_val)
else:
    np.savez('../data/cnn_dailymail.npz' , X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, X_val=X_val, y_val=y_val)


