import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from more_itertools import locate
import random as rand
import numpy as np

n_samples_to_plot = int(sys.argv[1])
results_file = sys.argv[2]

fin = open(results_file, 'r')
lines = [line.split() for line in fin.readlines()]
true = [float(line[1]) for line in lines if len(line) > 1]
preds = [float(line[2]) for line in lines if len(line) > 1 ]

sampling = rand.sample(range(len(preds)), k=n_samples_to_plot)
invalids = list(locate(true, lambda x: x == -1))
sampling = set(sampling).difference(set(invalids))
preds_samp = [preds[i] for i in sampling]
true_samp  = [true[i] for i in sampling]

#for p, v in zip(preds_samp, true_samp):
#	print(p, v)

plt.scatter(true_samp, preds_samp, 0.1)
plt.xlim(0, 3)
plt.ylim(0, 3)
plt.title('Regression on BERT-uncased')
plt.xlabel('True values')
plt.ylabel('Predicted values')
plt.savefig('plot.png')



correlation = np.corrcoef(preds, true)
print('correlation: ', correlation)
