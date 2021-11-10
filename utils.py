import csv
import numpy as np
import math

def read_csv(filename):
    dataset = list()
    try:
        with open(filename) as csvfile:
            reader = csv.reader(csvfile)
            try:
                for line in reader:
                    row = list()
                    flag = True
                    for value in line:
                        try:
                            value = float(value)
                        except:
                            if not value:
                                flag = False
                                value = np.nan
                        row.append(value)
                    dataset.append(row)
            except csv.Error as e:
                print(f'file {filename}, line {reader.line_num}: {e}')
    except Exception as e:
        print(f'Error with file: {e}')
        exit()
    return np.array(dataset, dtype=object)

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

def print_acceptable_features():
    print('''6 - Arithmancy
7 - Astronomy
8 - Herbology
9 - Defense Against the Dark Arts
10 - Divination
11 - Muggle Studies
12 - Ancient Runes
13 - History of Magic
14 - Transfiguration
15 - Potions
16 - Care of Magical Creatures
17 - Charms
18 - Flying''')
