import sys
import numpy as np
import math
from utils import *

def print_func_features(name_func, names, features, widths, f):
    print(name_func.ljust(7), end='')
    for i, feature_name in enumerate(names[6:]):
        line = '%.6f' % f(features[feature_name])
        print(line.center(widths[i]), end='  ')
    print('')


not_num_cols_ind = {1, 2, 3, 4, 5}
fin = open(sys.argv[1], "r")
names = np.array(fin.readline()[:-1].split(','))
features = dict()
for line in fin:
    for i, elem in enumerate(line[:-1].split(',')):
        if i not in not_num_cols_ind:
            if elem == '':
                elem = '0'
            elem = float(elem)
        if names[i] not in features.keys():
            features[names[i]] = [elem]
        else:
            features[names[i]].append(elem)
fin.close()
print(' ' * 7, end='')
widths = []
for name in names[6:]:
    w = len(name) if len(name) >= 13 else 13
    widths.append(w)
    print(name.center(w), end='  ')
print('')
print_func_features("Count", names, features, widths, count)
print_func_features("Mean", names, features, widths, mean)
print_func_features("Std", names, features, widths, std)
print_func_features("Min", names, features, widths, min_elem)
print_func_features("25%", names, features, widths, lambda x: percentile(x, 0.25))
print_func_features("50%", names, features, widths, lambda x: percentile(x, 0.5))
print_func_features("75%", names, features, widths, lambda x: percentile(x, 0.75))
print_func_features("Max", names, features, widths, max_elem)
