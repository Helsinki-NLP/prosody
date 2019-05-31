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
log_values = [np.log(value + 0.1) for value in values]

values.sort()
log_values.sort()

plt.scatter(range(0, len(values)), values, 0.1)
plt.scatter(range(0, len(values)), log_values, 0.1, c='r')

plt.title('prosody values vs log(values) - ' + file_name)
plt.ylabel('value')
plt.savefig('values_vs_log_' + file_name + '.png')


