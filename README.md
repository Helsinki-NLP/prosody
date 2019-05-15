# BERT-prosody
Prosody prediction using BERT

## Usage

To install the requirements run:

```console
pip3 install -r requirements.txt
```

To download the word embeddings for the LSTM model run:
```console
./download_embeddings.sh
```

For the **BERT** model run training by executing:

```console
# Train BERT-Uncased
python3 main.py \
    --model BertUncased \
    --batch_size 16 \
    --epochs 5 \
    --save_path results.txt \
    --log_every 50 \
    --learning_rate 0.00005 \
    --weight_decay 0 \
    --gpu 0 \
    --fraction_of_sentences 1 \
    --optimizer adam \
    --seed 1234
```

For the **LSTM** model run training by executing:
```console
# Train 3-layer BiLSTM
python3 main.py \
    --model LSTM \
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
And 0 0
those 2 2
who 0 0
meet 1 2
in 0 0
the 0 0
great 1 1
hall 1 1
with 0 0
the 0 0
white 2 1
Atlas 2 2
? NA NA
```

## Models

* [BERT](https://arxiv.org/abs/1810.04805)-base Uncased
* [BERT](https://arxiv.org/abs/1810.04805)-base Cased
* [Minitagger](https://github.com/karlstratos/minitagger) A multi-class SVM trained using GloVe word embeddings. Paper: https://www.aclweb.org/anthology/W15-1511
* 1-layer 600D LSTM
* 3-layer 600D Bidirectional LSTM

## Results

**Results (excluding NA tag)**

| Model             | accuracy    |precision   |  recall     |f1-score    |
| ---               | ---         | ---        | ---         | ---        |
| BERT-base cased   |  **0.6792** |  0.6754    |  **0.6792** | **0.6758** |
| BERT-base uncased |  0.6758     | **0.6854** |  0.6758     | 0.6756     |
| BiLSTM (3 layers) |  0.6472     |  0.6388    |  0.6472     | 0.6424     |
| LSTM (1 layers)   |  0.6376     |  0.6230    |  0.6376     | 0.6270     |
| Minitagger (SVM)  |  0.6455     |  0.6402    |  0.6455     | 0.6426     |
| Majority per word |  0.6180     |  0.6205    |  0.6205     | 0.6205     |
| Majority class    |  0.5087     |  0.2588    |  0.5087     | 0.3430     |


## Analysis

Sample analyses (to be reproduced for the paper)

![Bert-uncased](images/confusion_matrix-BertUncased.png)

|                 |precision   |  recall     |f1-score    |  support   |
| ---             | ---        | ---         | ---        | ---        |
|    label 0      |  0.8271    | 0.8131      | 0.8200     | 45818      |
|    label 1      | 0.4754     | 0.5883      | 0.5259     | 24168      |
|    label 2      | 0.6190     |  0.4660     | 0.5317     | 20087      |
| **avg / total** | **0.6863** |  **0.6754** | **0.6768** |  **90073** |

![WordMajority](images/confusion_matrix-WordMajority.png)

|                 |precision   |  recall     |f1-score    |  support   |
| ---             | ---        | ---         | ---        | ---        |
|     label 0     | 0.7298     | 0.8498      | 0.7852     | 45818      |
|     label 1     | 0.4419     | 0.4206      | 0.4310     | 24168      |
|     label 2     | 0.4946     | 0.3379      | 0.4015     | 20087      |
| **avg / total** | **0.6205** | **0.6205**  | **0.6205** | **90073** |


| Model             | Test acc (incl punctuation) | Test acc (no punctuation) |
| ---               |  ---                        | ---                       |
| BERT-base uncased | **72.5%**                   | **68.7%** (not able to reproduce) |
| BERT-base cased   | 70.4%                       | 65.8%                     |
| Minitagger (SVM)  | 69.8%                       | 65.6%                     |
| LSTM (1 layers)   | 69.2%                       | 63.6%                     |
| BiLSTM (3 layers) | 70.5%                       | 64.6%                     |
| Majority per word | 54.4%                       | 61.8%                     |
| Majority class    | 44.0%                       | 50.9%                     |


## TODO

* BERT (DONE)
* Words embeddings + LSTM (DONE)
* BERT + LSTM (Aarne)
* Regression (Ongoing)
* Majority for each word (DONE)
* Class-encodings for ordinal class labels (Hande)
* Context model (neighbours) (Hande)
* CRF (Hande)
* BERT + position encoding
* Other pre-trained models (GPT, ELMo etc)
* Sort sentences before forming batches? (Hande)
