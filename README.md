# BERT-prosody
Prosody prediction using BERT

## Usage

To install the requirements run:

```console
pip3 install -r requirements.txt
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
    --number_of_sents 33047 \ # Use this to select only a subset of examples
    --test_and_dev_split .1 \ # This will create splits 80/10/10
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

## Results


| Model             | Training data | Dev data | Test data    | Epochs | Test acc |
| ---               | ---           | ---      | ---          | ---    | ---      |
| BERT-base uncased | 26437         | 3305     | 3305         | 2      | 73.38%   |
| BERT-base cased   | 26437         | 3305     | 3305         | 2      | 72.48%   |

## Experiments

* BERT (DONE)
* Words embeddings + LSTM (Aarne)
* BERT + LSTM (Aarne)
* Majority for each word (Hande)
* Context model (neighbours) (Hande)
* CRF (Hande)
* BERT + position encoding
* Other pre-trained models (GPT, ELMo etc)