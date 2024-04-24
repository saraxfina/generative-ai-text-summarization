import numpy as np

data = np.load('../data/cnn_dailymail.npz', allow_pickle=True)
#data = np.load('../data/cnn_dailymail_small.npz', allow_pickle=True)

X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']
X_val = data['X_val']
y_val = data['y_val']

data = np.load('../data/news_api_data.npz', allow_pickle=True)
#data = np.load('../data/news_api_data_small.npz', allow_pickle=True)

X_train = np.concatenate((data['X_train'], X_train))
y_train = np.concatenate((data['y_train'], y_train))
X_test = np.concatenate((data['X_test'], X_test))
y_test = np.concatenate((data['y_test'], y_test))
X_val = np.concatenate((data['X_val'], X_val))
y_val = np.concatenate((data['y_val'], y_val))

np.savez('../data/dataset.npz' , X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, X_val=X_val, y_val=y_val)
