import datasets 
from datasets import load_dataset
import numpy as np 

dataset = load_dataset("cnn_dailymail", '3.0.0')

train = dataset['train']
test = dataset['test']
val = dataset['validation']

train_small = train[:200]
val_small = val[:40]
test_small = test[:40]

np.savez('../data/cnn_dailymail.npz' , train=train, test=test, val=val)

np.savez('../data/cnn_dailymail_small.npz', train=train_small, test=test_small, val=val_small)
