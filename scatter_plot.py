import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import read_csv

if len(sys.argv) < 4:
    print("Usage: python histogram.py dataset_name num_of_col1 num_of_col2")
num_col1 = int(sys.argv[2])
num_col2 = int(sys.argv[3])
if num_col1 < 6 or num_col1 > 18 or num_col2 < 6 or num_col2 > 18:
    print("num_of_col must be between 6 and 18")

data = read_csv(sys.argv[1])
names = data[0]
data = data[1:, :]

plt.scatter(data[:, num_col1], data[:, num_col2])
plt.xlabel(names[num_col1])
plt.ylabel(names[num_col2])
plt.show()
