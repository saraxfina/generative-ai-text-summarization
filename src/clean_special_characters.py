import numpy as np
import re

data = np.load('../data/dataset.npz', allow_pickle=True)

X_train = data['X_train'].tolist()
y_train = data['y_train'].tolist()
X_test = data['X_test'].tolist()
y_test = data['y_test'].tolist()
X_val = data['X_val'].tolist()
y_val = data['y_val'].tolist()

X_train = [re.sub(r'\r\n|\r|\n', ' ', _) for _ in X_train]
X_train = [re.sub('[^a-zA-Z0-9 ]+', '', _) for _ in X_train]

y_train = [re.sub(r'\r\n|\r|\n', ' ', _) for _ in y_train]
y_train = [re.sub('[^a-zA-Z0-9 ]+', '', _) for _ in y_train]

X_test = [re.sub(r'\r\n|\r|\n', ' ', _) for _ in X_test]
X_test = [re.sub('[^a-zA-Z0-9 ]+', '', _) for _ in X_test]

y_test = [re.sub(r'\r\n|\r|\n', ' ', _) for _ in y_test]
y_test = [re.sub('[^a-zA-Z0-9 ]+', '', _) for _ in X_test]

X_val = [re.sub(r'\r\n|\r|\n', ' ', _) for _ in X_val]
X_val = [re.sub('[^a-zA-Z0-9 ]+', '', _) for _ in X_val]

y_val = [re.sub(r'\r\n|\r|\n', ' ', _) for _ in y_val]
y_val = [re.sub('[^a-zA-Z0-9 ]+', '', _) for _ in y_val]

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
X_test = np.asarray(X_test)
y_test = np.asarray(y_test)
X_val = np.asarray(X_val)
y_val = np.asarray(y_val)

np.savez('../data/dataset_cleaned.npz' , X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, X_val=X_val, y_val=y_val)
