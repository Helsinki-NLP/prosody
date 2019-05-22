
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

| train | label | features              | accuracy |
|-------|-------|-----------------------|----------|
| 360   | l2    | unigram               | 66.988   |
| 360   | l2    | bigram                | 68.768   |
| 360   | l2    | trigram               | 69.634   |
| 360   | l2    | ud3.trigram           | 69.333   |
| 360   | l2    | ud4.trigram           | 65.105   |
| 360   | l2    | ud5.trigram           | 65.268   |
| 360   | l2    | ud2+ud3+ud4.trigram   | 69.820   |


* Predicting boundary labels (3 classes + NA):

| train | label | features              | accuracy |
|-------|-------|-----------------------|----------|
| 360   | l3    | unigram               | 67.132   |
| 360   | l3    | bigram                | 67.937   |
| 360   | l3    | trigram               | 76.039   |
| 360   | l3    | ud3.trigram           | 75.953   |
| 360   | l3    | ud4.trigram           | 74.303   |
| 360   | l3    | ud5.trigram           | 74.416   |
| 360   | l3    | ud2+ud3+ud4.trigram   | 76.076   |


* Predicting combined labels (9 classes + NA):

| train | label | features              | accuracy |
|-------|-------|-----------------------|----------|
| 360   | l23   | unigram               |          |
| 360   | l23   | bigram                | 49.603   |
| 360   | l23   | trigram               | 54.373   |
| 360   | l23   | ud3.trigram           | 54.145   |
| 360   | l23   | ud4.trigram           | 50.279   |
| 360   | l23   | ud5.trigram           | 50.443   |
| 360   | l23   | ud2+ud3+ud4.trigram   | 54.574   |
