import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import read_csv
import warnings
warnings.filterwarnings("ignore")

if len(sys.argv) < 3:
    print("Usage: python histogram.py dataset_name [all or numbers of fetures]")
if sys.argv[2] == 'all':
    features = np.arange(6, 19)
else:
    try:
        features = np.array(list(map(int, sys.argv[2:])))
    except:
        print("Usage: python histogram.py dataset_name [all or numbers of fetures]")
for num_col in features:
    if num_col < 6 or num_col > 18:
        print("num of feature must be between 6 and 18")
        exit()
data = read_csv(sys.argv[1])
names = data[0]
data = data[1:, :]
data = data[data[:, 1].argsort()]

labels = ['Grynffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
size = len(features)
fig, axes = plt.subplots(nrows=size, ncols=size)
for i in range(size):
    for j in range(size):
        if i == j:
            axes[i, j].hist(data[:327, features[i]], color='red', alpha=0.5)
            axes[i, j].hist(data[327:856, features[i]], color='yellow', alpha=0.5)
            axes[i, j].hist(data[856:1299, features[i]], color='blue', alpha=0.5)
            axes[i, j].hist(data[1299:, features[i]], color='green', alpha=0.5)
        else:
            axes[i, j].scatter(data[:, features[j]], data[:, features[i]])
        if i == size - 1:
            axes[i, j].set_xlabel(names[features[j]])
        if j == 0:
            axes[i, j].set_ylabel(names[features[i]])
plt.legend(labels, loc='best', frameon=False)
plt.show()
