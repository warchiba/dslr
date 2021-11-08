import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import read_csv
import warnings
warnings.filterwarnings("ignore")

if len(sys.argv) < 3:
    print("Usage: python histogram.py dataset_name num_of_col")
num_col = int(sys.argv[2])
if num_col < 6 or num_col > 18:
    print("num_of_col must be between 6 and 18")

data = read_csv(sys.argv[1])
names = data[0]
data = data[1:, :]
data = data[data[:, 1].argsort()]

labels = ['Grynffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

plt.hist(data[:327, num_col], color='red', alpha=0.5)
plt.hist(data[327:856, num_col], color='yellow', alpha=0.5)
plt.hist(data[856:1299, num_col], color='blue', alpha=0.5)
plt.hist(data[1299:, num_col], color='green', alpha=0.5)
plt.legend(labels, loc='best')
plt.title(names[num_col])
plt.xlabel("Marks")
plt.ylabel("Number of students")
plt.show()

#print(data)
