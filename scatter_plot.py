import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import read_csv, print_acceptable_features

if len(sys.argv) < 4:
    print("Usage: python histogram.py dataset_name num_of_col1 num_of_col2")
    print_acceptable_features()
    exit()
num_col1 = int(sys.argv[2])
num_col2 = int(sys.argv[3])
if num_col1 < 6 or num_col1 > 18 or num_col2 < 6 or num_col2 > 18:
    print("num_of_col must be between 6 and 18")
    print_acceptable_features()
    exit()

data = read_csv(sys.argv[1])
names = data[0]
data = data[1:, :]
data = data[data[:, 1].argsort()]

labels = ['Grynffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

plt.scatter(data[:327, num_col1], data[:327, num_col2], c='red')
plt.scatter(data[327:856, num_col1], data[327:856, num_col2], c='yellow', marker=',')
plt.scatter(data[856:1299, num_col1], data[856:1299, num_col2], c='blue', marker='^')
plt.scatter(data[1299:, num_col1], data[1299:, num_col2], c='green', marker='P')
plt.xlabel(names[num_col1])
plt.ylabel(names[num_col2])
plt.legend(labels, loc='best')
plt.show()
