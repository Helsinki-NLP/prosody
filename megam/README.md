
# Classification experiments with megam

Software:

* megam: http://users.umiacs.umd.edu/~hal/megam/version0_3
* udpipe: http://ufal.mff.cuni.cz/udpipe

The experiments are run using the Makefile in this directory and various variables can be set to adjust the setup, features and data sets. Right now, there are binaries included for Mac OSX. If you want to use the software compiled on your machine then you need to set MEGAM and UDPIPE in the Makefile. Here are some other important variables:

* FEAT: features to be used (unigram, bigram, trigram)
* TRAIN: training data to be used (100 or 360)
* LABEL: label to be predicted (2=prosody, 3=boundary, 23=combined)

NA is treated as label '0' and the NA tag is revovered in some simple post-processing step that replaces the predicted label with NA for all punctuations in the data. 

The variables that manipulate the experiments can be set directly when calling the makefile. Here are some example calls:

```
make FEAT=unigram TRAIN=100 eval
make FEAT=bigram TRAIN=360 eval
# make FEAT=trigram TRAIN=360 LABEL=23 eval
```

The mode can also use annotation coming from UDpipe. In that case, the FEAT variable needs to start with the prefix `udX.` where X is the column to be used from the UD annotation file. Note that UDTAG needs to be set to the same column number. Here some examples:

```
make UDTAG=3 FEAT=ud3.trigram TRAIN=100 eval
make UDTAG=4 FEAT=ud4.bigram TRAIN=100 eval
make UDTAG=4 FEAT=ud4.trigram TRAIN=360 eval
```

There is also a target for a specific combination of annotation features (ud2+ud3+ud4), which corresponds to surface words, lemmas and universal POS tags. To run with trigrams of those features, run:

```
make FEAT=ud2+ud3+ud4.trigram TRAIN=360 eval
```

## Results

All experiments use 25 iterations and multiclass average perceptron as the model. The detailed results with class-based precision and recall values are in the files `*.eval`. Below is a summary of accuracy scores.


* Predicting prosody (3 classes + NA):

| train | label | features              | accuracy | without NA | binary |
|-------|-------|-----------------------|----------|------------|--------|
| 360   | l2    | unigram               | 66.988   | 62.585     | 78.465 |
| 360   | l2    | bigram                | 68.768   | 64.627     | 79.062 |
| 360   | l2    | trigram               | 69.634   | 65.616     | 80.167 |
| 360   | l2    | ud3.trigram           | 69.333   | 65.273     | 80.481 |
| 360   | l2    | ud4.trigram           | 65.105   | 60.437     | 76.816 |
| 360   | l2    | ud5.trigram           | 65.268   | 60.622     | 76.612 |
| 360   | l2    | ud2+ud3+ud4.trigram   | 69.820   | 65.824     | 




* Predicting boundary labels (3 classes + NA):

| train | label | features              | accuracy | without NA |
|-------|-------|-----------------------|----------|------------|
| 360   | l3    | unigram               | 67.132   | 62.793     |
| 360   | l3    | bigram                | 67.937   | 63.708     |
| 360   | l3    | trigram               | 76.039   | 72.949     |
| 360   | l3    | ud3.trigram           | 75.953   | 72.852     |
| 360   | l3    | ud4.trigram           | 74.303   | 70.967     |
| 360   | l3    | ud5.trigram           | 74.416   | 71.100     |
| 360   | l3    | ud2+ud3+ud4.trigram   | 76.076   | 72.988     |



* Predicting combined labels (9 classes + NA):

| train | label | features              | accuracy | without NA |
|-------|-------|-----------------------|----------|------------|
| 360   | l23   | unigram               |          | 40.966     |
| 360   | l23   | bigram                | 49.603   | 42.742     |
| 360   | l23   | trigram               | 54.373   | 48.182     |
| 360   | l23   | ud3.trigram           | 54.145   | 47.930     |
| 360   | l23   | ud4.trigram           | 50.279   | 43.511     |
| 360   | l23   | ud5.trigram           | 50.443   | 43.698     |
| 360   | l23   | ud2+ud3+ud4.trigram   | 54.574   | 48.413     |

test.360.l23.bigram.multitron.eval-no-NA: accuracy 42.742 (38495/90063)
test.360.l23.trigram.multitron.eval-no-NA:	   accuracy	48.182 (43394/90063)
test.360.l23.ud3.trigram.multitron.eval-no-NA:	   accuracy	47.930 (43167/90063)
test.360.l23.ud4.trigram.multitron.eval-no-NA:	   accuracy	43.511 (39187/90063)
test.360.l23.ud5.trigram.multitron.eval-no-NA:	   accuracy	43.698 (39356/90063)
test.360.l23.unigram.multitron.eval-no-NA:	   accuracy	40.966 (36895/90063)



* If we know about the boundary labels and predict prosody labels:

| train | label | features              | accuracy | without NA |
|-------|-------|-----------------------|----------|------------|
| 360   | l2    | l3.unigram            | 59.967   | 54.555     |
| 360   | l2    | l3.bigram             | 61.208   | 55.980     |
| 360   | l2    | l3.trigram            | 61.509   | 56.336     |
| 360   | l2    | ud2+l3.trigram        | 71.181   | 67.396     |

