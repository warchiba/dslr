import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
from utils import read_csv, print_acceptable_features
import warnings
warnings.filterwarnings("ignore")

if len(sys.argv) < 3:
    print("Usage: python histogram.py dataset_name [all or numbers of fetures]")
    print_acceptable_features()
    exit()
if sys.argv[2] == 'all':
    features = np.arange(6, 19)
else:
    try:
        features = np.array(list(map(int, sys.argv[2:])))
    except:
        print("Usage: python histogram.py dataset_name [all or numbers of fetures]")
        print_acceptable_features()
        exit()
for num_col in features:
    if num_col < 6 or num_col > 18:
        print("num of feature must be between 6 and 18")
        print_acceptable_features()
        exit()
data = read_csv(sys.argv[1])
names = data[0]
data = data[1:, :]
data = data[data[:, 1].argsort()]
labels = ['Grynffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
size = len(features)
font = {'family' : 'DejaVu Sans',
          'weight' : 'light',
          'size'   : 7}
matplotlib.rc('font', **font)
fig, axes = plt.subplots(nrows=size, ncols=size)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
for i in range(size):
    for j in range(size):
        if i == j:
            axes[i, j].hist(data[:327, features[i]], color='red', alpha=0.5)
            axes[i, j].hist(data[327:856, features[i]], color='yellow', alpha=0.5)
            axes[i, j].hist(data[856:1299, features[i]], color='blue', alpha=0.5)
            axes[i, j].hist(data[1299:, features[i]], color='green', alpha=0.5)
        else:
            axes[i, j].scatter(data[:327, features[j]], data[:327, features[i]], c='red', marker='.', alpha=0.5)
            axes[i, j].scatter(data[327:856, features[j]], data[327:856, features[i]], c='yellow', marker='.', alpha=0.5)
            axes[i, j].scatter(data[856:1299, features[j]], data[856:1299, features[i]], c='blue', marker='.', alpha=0.5)
            axes[i, j].scatter(data[1299:, features[j]], data[1299:, features[i]], c='green', marker='.', alpha=0.5)
        if i == size - 1:
            axes[i, j].set_xlabel(names[features[j]].replace(' ', '\n'))
        else:
            axes[i, j].tick_params(labelbottom=False)
        if j == 0:
            axes[i, j].set_ylabel(names[features[i]].replace(' ', '\n'))
        else:
            axes[i, j].tick_params(labelleft=False)
        axes[i, j].spines['right'].set_visible(False)
        axes[i, j].spines['top'].set_visible(False)
plt.legend(labels, loc='best', frameon=False, bbox_to_anchor=(1, 0.5))
plt.show()
