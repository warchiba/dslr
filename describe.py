import sys
import numpy as np
import math


def count(lst):
    return len(lst)


def mean(lst):
    s = 0
    for el in lst:
        s += el
    return s / len(lst)


def std(lst):
    m = mean(lst)
    s = 0
    for el in lst:
        s += (el - m)**2
    return math.sqrt(s / (len(lst) - 1))


def min_elem(lst):
    m = lst[0]
    for el in lst:
        if m > el:
            m = el
    return m


def max_elem(lst):
    m = lst[0]
    for el in lst:
        if m < el:
            m = el
    return m


def sort_lst(lst):
    swapped = False
    for i in range(len(lst) - 1, 0, -1):
        for j in range(i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                swapped = True
        if swapped:
            swapped = False
        else:
            break
    return lst


def percentile(lst, percent):
    lst = sort_lst(lst)
    k = (len(lst) - 1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return lst[int(k)]
    d0 = lst[int(f)] * (c - k)
    d1 = lst[int(c)] * (k - f)
    return d0+d1


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
            #print(names[i])
            #print(features[names[i]])
            features[names[i]].append(elem)
fin.close()
print(' ' * 7, end='')
widths = []
for name in names[6:]:
    w = len(name) if len(name) >= 11 else 11
    widths.append(w)
    print(name.center(w), end='  ')
print('\n')
print('Count  ', end='')
for i, feature_name in enumerate(names[6:]):
    line = '%.6f' % count(features[feature_name])
    print(line.center(widths[i]), end='  ')
print('\n')
print('Min    ', end='')
for i, feature_name in enumerate(names[6:]):
    line = '%.6f' % min_elem(features[feature_name])
    print(line.center(widths[i]), end='  ')
print('\n')
print('Mean   ', end='')
for i, feature_name in enumerate(names[6:]):
    line = '%.6f' % mean(features[feature_name])
    print(line.center(widths[i]), end='  ')
print('\n')



