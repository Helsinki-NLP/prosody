import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from more_itertools import locate
import random as rand
import numpy as np

data_file = sys.argv[1]
base_name, file_name = os.path.split(data_file)
file_name = file_name.split('.')[0]

fin = open(data_file, 'r')
lines = [line.split() for line in fin.readlines()]
values = [float(line[3]) for line in lines if len(line) > 1 and line[3] != 'NA']

bins = np.arange(0, 4, 0.1)

plt.hist(values, bins)

plt.title('prosody values histogram [' + file_name + ']')
plt.ylabel('prosody value')
plt.savefig('values_histogram_' + file_name + '.png')


