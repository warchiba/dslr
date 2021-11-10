import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import read_csv
from logreg import MyLogisticRegression
from sklearn.metrics import accuracy_score

if len(sys.argv) < 3:
    print("Usage: python histogram.py dataset_name weights_name")
    exit()

data = read_csv(sys.argv[1])
names = data[0]
data = data[1:, :]
X = np.delete(data[:, 6:], 1, 1).astype('float32')
np.place(X, np.isnan(X), 0)
X_test = np.array((X - X.mean(axis=0)) / X.std(axis=0), dtype='float32')
y_test = data[:, 1]

lrG = MyLogisticRegression()
lrH = MyLogisticRegression()
lrR = MyLogisticRegression()
lrS = MyLogisticRegression()

try:
    with open(sys.argv[2], 'r') as fin:
        w = np.array(list(map(float, fin.readline().split(','))))
        lrG.set_weights(w)
        w = np.array(list(map(float, fin.readline().split(','))))
        lrH.set_weights(w)
        w = np.array(list(map(float, fin.readline().split(','))))
        lrR.set_weights(w)
        w = np.array(list(map(float, fin.readline().split(','))))
        lrS.set_weights(w)
except Exception as e:
    print(f'Error with file: {e}')
    exit()

y_pred = np.array(np.argmax([lrG.predict_proba(X_test), lrH.predict_proba(X_test), lrR.predict_proba(X_test), lrS.predict_proba(X_test)], axis=0), dtype='object')
np.place(y_pred, y_pred == 0, 'Gryffindor')
np.place(y_pred, y_pred == 1, 'Hufflepuff')
np.place(y_pred, y_pred == 2, 'Ravenclaw')
np.place(y_pred, y_pred == 3, 'Slytherin')

with open('houses.csv', 'w') as fout:
    print("Index,Hogwarts House", file=fout)
    for i, elem in enumerate(y_pred):
        print(i, elem, sep=',', file=fout)
