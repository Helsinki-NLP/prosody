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

Run training by executing:

```console
python3 main.py \
    --model BertUncased \
    --batch_size 16 \
    --epochs 5 \
    --save_path results.txt \
    --log_every 50 \
    --learning_rate 0.00005 \
    --weight_decay 0 \
    --gpu 0 \
    --fraction_of_sentences 1 \ # Use this to select only a subset of examples (float)
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

* BERT-base Uncased
* BERT-base Cased
* [Minitagger](https://github.com/karlstratos/minitagger) SVM trained using GloVe word embeddings. Paper: https://www.aclweb.org/anthology/W15-1511

## Results


| Model             | Test acc (incl punctuation) |Test acc (no punctuation) |
| ---               |  ---                        | ---                      |
| BERT-base uncased | 72.5%                       | 68.7%                    |
| BERT-base cased   |                             |                          |
| Minitagger (SVM)  | 69.8%                       | 65.6%                    |

## Experiments

* BERT (DONE)
* Words embeddings + LSTM (Aarne)
* BERT + LSTM (Aarne)
* Regression (Ongoing)
* Majority for each word (Hande)
* Class-encodings for ordinal class labels (Hande)
* Context model (neighbours) (Hande)
* CRF (Hande)
* BERT + position encoding
* Other pre-trained models (GPT, ELMo etc)
