import numpy as np
import matplotlib.pyplot as plt
import sys
from utils import read_csv

if len(sys.argv) < 2:
    print("Usage: python histogram.py dataset_name")

data = read_csv(sys.argv[1])
names = data[0]
data = data[1:, :]
