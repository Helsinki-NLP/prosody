# Predicting Prosodic Prominence from Text
This repository contains the prosody corpus and the code from the forthcoming [NoDaLiDa](https://nodalida2019.org/) paper: [Predicting Prosodic Prominence from Text with Pre-trained Contextualized Word Representations]()
* Helsinki Prosody Corpus
* System for predicting prosodic prominence

If you use the corpus or the system, please cite: 

```
@inproceedings{helsinki_prosody_2019,
  author = {Aarne Talman and Antti Suni and Hande Celikkanat and Sofoklis Kakouros and J\"org Tiedemann and Martti Vainio},
  title = {Predicting Prosodic Prominence from Text with Pre-trained Contextualized Word Representations},
  booktitle = {Proceedings of NoDaLiDa},
  year = {2019}
}
```

## Corpus

This repository contains the largest annotated dataset with labels for prosodic prominence. The corpus is available in the [data](https://github.com/Helsinki-NLP/prosody/tree/master/data) folder.  

### Corpus statistics

| Datasesets    |  Speakers  |  Sentences  |  Words     |  Label: 0  |  Label: 1 |  Label: 2 |
| ---           | ---        | ---         | ---        | ---        | ---       | ---       |
| train-100     |  247       |   33,041    |  570,592   |  274,184   |  155,849  |  140,559  |
| train-360     |  904       |  116,262    |  2,076,289 |  1,003,454 |  569,769  |  503,066  |
| dev           |  40        |  5726       |  99,200    |  47,535    |  27,454   |  24,211   |
| test          |  39        |  4821       |  90,063    |  43,234    |  24,543   |  22,286   |
| **Total:**        |  **1230**      |  **159,850**    |  **2,836,144** |  **1,368,407** |  **777,615**  |  **690,122**  |

## Usage of the system

To use the system following dependencies need to be installed:

* Python 3
* torch>=1.0
* argparse
* pytorch_transformers
* numpy
* matplotlib (only used for visualizations)
* sklearn (only used for prediction metrics)


To install the requirements run:

```console
pip3 install -r requirements.txt
```

To download the word embeddings for the LSTM model run:
```console
./download_embeddings.sh

```

### Models included:
* BERT
* LSTM
* Majority class per word
* See *model.py* for the complete list

For the **BERT** model run training by executing:

```console
# Train BERT-Uncased
python3 main.py \
    --model BertUncased \
    --train_set train_360 \
    --batch_size 32 \
    --epochs 2 \
    --save_path results.txt \
    --log_every 50 \
    --learning_rate 0.00005 \
    --weight_decay 0 \
    --gpu 0 \
    --fraction_of_sentences 1 \
    --optimizer adam \
    --seed 1234
```

For the **Bidirectional LSTM** model run training by executing:
```console
# Train 3-layer BiLSTM
python3 main.py \
    --model BiLSTM \
    --train_set train_360 \
    --layers 3 \
    --hidden_dim 600 \
    --batch_size 64 \
    --epochs 5 \
    --save_path results.txt \
    --log_every 50 \
    --learning_rate 0.001 \
    --weight_decay 0 \
    --gpu 0 \
    --fraction_of_sentences 1 \
    --optimizer adam \
    --seed 1234
```


## Output

Output of the system is a text file with the following structure:

```
<word> tab <label> tab <prediction>
```

Example output:
```
And    0     0
those  2     2
who    0     0
meet   1     2
in     0     0
the    0     0
great  1     1
hall   1     1
with   0     0
the    0     0
white  2     1
Atlas  2     2
?      NA    NA
```

## Results

Main experimental results from the paper using the *test360* dataset.

|    Model                 |  Test accuracy (2-way)  |  Test accuracy (3-way) |
| ---                      | ---                     | ---                    |
| BERT-base                |  **83.2%**                  |  **68.6%**                 |
| 3-layer BiLSTM           |  82.1%                  |  66.4%                 | 
| CRF ([MarMoT](http://cistern.cis.lmu.de/marmot/)) |  81.8%                  |  66.4%                 |
| SVM+GloVe ([Minitagger](https://github.com/karlstratos/minitagger))  |  80.8%                  |  65.4%                 |
| Majority class per word  |  80.2%                  |  62.4%                 |
| Majority class           |  52.0%                  |  48.0%                 |
| Random                   |  49.0%                  |  39.5%                 |


