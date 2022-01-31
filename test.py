import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score, f1_score


# a = np.random.randn(64, 4, 5)
# print(a[:2].shape)
# print(a[2:].shape)
# print(a.shape)
# n = a.ndim
# print(a.transpose((*range(n-2), n-1, n-2)).shape)
# b = np.random.randn(3, 4)
# r = b @ a
# print(r.shape)

# c = np.reshape(b, (-1, 1))
# print(c.shape)
# c = np.reshape(c, b.shape)
# print((b == c).all())

# data = pd.read_csv('sir-toy-dataset/trainNN.txt', delim_whitespace=True, header=None)
#
# print(data.iloc[:, 4].unique())
#
# x_train = np.array(data.iloc[:, :-1])
# y_train = np.array(data.iloc[:, -1])
#
# y_train = pd.get_dummies(y_train)
# print(y_train.head())
# print(np.array(y_train.head()))

# print(x_train.shape)
# print(y_train.shape)

# x_train = np.expand_dims(x_train, axis=2)
# y_train = np.expand_dims(y_train, axis=2)

# print(x_train.shape)
# print(y_train.shape)

# data_size = x_train.shape[0]
# batch_size = 100
# iteration = data_size // batch_size

# for i in range(iteration):
#     x_train_batch = x_train[i:i+batch_size]
#     y_train_batch = y_train[i:i+batch_size]
#     print(x_train_batch.shape, end=' ')
#     print(y_train_batch.shape)

# a = np.random.rand(5, 3, 2, 4)
# b = np.reshape(a, (-1, 1))
# print(a.shape)
# print(b.shape)

def safe_log(x):
    x[x == 0] = 1
    return np.log(x)


def cross_entropy_loss(y_true, y_pred):
    return np.average(np.sum(-y_true * safe_log(y_pred), axis=1))


y_true = np.random.rand(64, 4, 1)
y_pred = np.random.rand(64, 4, 1)
print(-y_true * safe_log(y_pred))
print(np.sum(-y_true * safe_log(y_pred), axis=1))
print(np.average(np.sum(-y_true * safe_log(y_pred), axis=1)))
print(cross_entropy_loss(y_true, y_pred))

pred = np.argmax(y_true)
print(log_loss())