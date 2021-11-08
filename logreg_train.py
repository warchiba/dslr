import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import read_csv
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

def logit(x, w):
    return np.array(np.dot(x, w),dtype=np.float32)

def sigmoid(h):
    return 1. / (1 + np.exp(-h))

def generate_batches(X, y, batch_size):
    np.random.seed(42)
    X = np.array(X)
    y = np.array(y)
    perm = np.random.permutation(len(X))
    for batch_start in range(0, (len(X) // batch_size) * batch_size, batch_size):
        batch_X = X[perm[batch_start: batch_start + batch_size]]
        batch_y = y[perm[batch_start: batch_start + batch_size]]
        yield batch_X, batch_y


class MyLogisticRegression(object):
    def __init__(self):
        self.w = None

    def fit(self, X, y, epochs=20, lr=0.1, batch_size=100):
        n, k = X.shape
        if self.w is None:
            np.random.seed(42)
            self.w = np.random.randn(k + 1)

        X_train = np.concatenate((np.ones((n, 1)), X), axis=1)
        losses = []
        for i in range(epochs):
            for X_batch, y_batch in generate_batches(X_train, y, batch_size):
                predictions = self._predict_proba_internal(X_batch)
                loss = self.__loss(y_batch, predictions)
                losses.append(loss)
                self.w -= lr * self.get_grad(X_batch, y_batch, self._predict_proba_internal(X_batch))

        return losses

    def get_grad(self, X_batch, y_batch, predictions):
        grad_basic = X_batch.T @ (predictions - y_batch)
        return grad_basic

    def predict_proba(self, X):
        n, k = X.shape
        X_ = np.concatenate((np.ones((n, 1)), X), axis=1)
        return sigmoid(logit(X_, self.w))

    def _predict_proba_internal(self, X):
        """
        Возможно, вы захотите использовать эту функцию вместо predict_proba, поскольку
        predict_proba конкатенирует вход с вектором из единиц, что не всегда удобно
        для внутренней логики вашей программы
        """
        return sigmoid(logit(X, self.w))

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold

    def get_weights(self):
        return self.w.copy()

    def __loss(self, y, p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


if len(sys.argv) < 2:
    print("Usage: python histogram.py dataset_name")

data = read_csv(sys.argv[1])
names = data[0]
data = data[1:, :]

X = data[:, 6:].astype('float32')
y = data[:, 1]

# 300 последних оставляем на тест
X_train = np.array((X[:-300] - X[:-300].mean(axis=0)) / X[:-300].std(axis=0), dtype='float32')
y_train = y[:-300]
X_test = np.array((X[-300:] - X[-300:].mean(axis=0)) / X[-300:].std(axis=0), dtype='float32')
y_test = y[-300:]


lrG = MyLogisticRegression()
lrG.fit(X_train, np.array(y_train == 'Gryffindor', dtype='int'))
lrH = MyLogisticRegression()
lrH.fit(X_train, np.array(y_train == 'Hufflepuff', dtype='int'))
lrR = MyLogisticRegression()
lrR.fit(X_train, np.array(y_train == 'Ravenclaw', dtype='int'))
lrS = MyLogisticRegression()
lrS.fit(X_train, np.array(y_train == 'Slytherin', dtype='int'))
# ТУТ ДОЛЖНА БЫТЬ ЗАПИСЬ В ФАЙЛЫ, ЗАПИСЬ ВЕСОВ 

# y_pred = np.array(np.argmax([lrG.predict_proba(X_test), lrH.predict_proba(X_test), lrR.predict_proba(X_test), lrS.predict_proba(X_test)], axis=0))
# np.place(y_test, y_test == 'Gryffindor', 0)
# np.place(y_test, y_test == 'Hufflepuff', 0)
# np.place(y_test, y_test == 'Ravenclaw', 0)
# np.place(y_test, y_test == 'Slytherin', 0)

print(accuracy_score(lrG.predict_proba(X_test) > 0.5, np.array(y_test == 'Gryffindor', dtype='int')))
print(accuracy_score(lrH.predict_proba(X_test) > 0.5, np.array(y_test == 'Hufflepuff', dtype='int')))
print(accuracy_score(lrR.predict_proba(X_test) > 0.5, np.array(y_test == 'Ravenclaw', dtype='int')))
print(accuracy_score(lrS.predict_proba(X_test) > 0.5, np.array(y_test == 'Slytherin', dtype='int')))
