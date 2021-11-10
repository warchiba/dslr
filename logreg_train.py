import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import read_csv
from sklearn.metrics import accuracy_score
from logreg import MyLogisticRegression
import warnings
warnings.filterwarnings("ignore")

if len(sys.argv) < 2:
    print("Usage: python histogram.py dataset_name")
    exit()
data = read_csv(sys.argv[1])
names = data[0]
data = data[1:, :]
X = np.delete(data[:, 6:], 1, 1).astype('float32')
y = data[:, 1]
np.place(X, np.isnan(X), 0)
edge = 300
X_train = np.array((X[:-edge] - X[:-edge].mean(axis=0)) / X[:-edge].std(axis=0), dtype='float32')
y_train = y[:-edge]
X_test = np.array((X[-edge:] - X[-edge:].mean(axis=0)) / X[-edge:].std(axis=0), dtype='float32')
y_test = y[-edge:]
lrG = MyLogisticRegression()
lrG.fit(X_train, np.array(y_train == 'Gryffindor', dtype='int'))
lrH = MyLogisticRegression()
lrH.fit(X_train, np.array(y_train == 'Hufflepuff', dtype='int'))
lrR = MyLogisticRegression()
lrR.fit(X_train, np.array(y_train == 'Ravenclaw', dtype='int'))
lrS = MyLogisticRegression()
lrS.fit(X_train, np.array(y_train == 'Slytherin', dtype='int'))

asG = accuracy_score(lrG.predict_proba(X_test) > 0.5, np.array(y_test == 'Gryffindor', dtype='int'))
asH = accuracy_score(lrH.predict_proba(X_test) > 0.5, np.array(y_test == 'Hufflepuff', dtype='int'))
asR = accuracy_score(lrR.predict_proba(X_test) > 0.5, np.array(y_test == 'Ravenclaw', dtype='int'))
asS = accuracy_score(lrS.predict_proba(X_test) > 0.5, np.array(y_test == 'Slytherin', dtype='int'))
print('Accuracy for Gryffindor:', asG)
print('Accuracy for Hufflepuff:', asH)
print('Accuracy for Ravenclaw:', asR)
print('Accuracy for Slytherin:', asS)
print('Mean accuracy:', np.mean([asG, asH, asR, asS]))

with open('weights.csv', 'w') as fout:
    print(*lrG.get_weights(), sep=',', file=fout)
    print(*lrH.get_weights(), sep=',', file=fout)
    print(*lrR.get_weights(), sep=',', file=fout)
    print(*lrS.get_weights(), sep=',', file=fout)
