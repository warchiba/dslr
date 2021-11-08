import csv
import numpy as np
import math

def read_csv(filename):
  dataset = list()
  with open(filename) as csvfile:
    reader = csv.reader(csvfile)
    try:
      for line in reader:
        row = list()
        for value in line:
          try:
            value = float(value)
          except:
            if not value:
              value = 0
          row.append(value)
        dataset.append(row)
    except csv.Error as e:
      print(f'file {filename}, line {reader.line_num}: {e}')
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
