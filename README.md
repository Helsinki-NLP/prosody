# BERT-prosody
Prosody prediction using BERT

## Instructions

To install the requirements run:

```console
pip3 install -r requirements.txt
```

Run training by executing:

```console
python3 main.py \
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
