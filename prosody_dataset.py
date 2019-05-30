import os
import sys
import random
from torch.utils import data
from pytorch_pretrained_bert import BertTokenizer
import torch
import numpy as np

class Dataset(data.Dataset):
    def __init__(self, tagged_sents, tag_to_index, config, word_to_embid=None):
        sents, tags_li,values_li = [], [], [] # list of lists
        self.config = config

        for sent in tagged_sents:
            words = [word_tag[0] for word_tag in sent]
            tags = [word_tag[1] for word_tag in sent]
            values = [word_tag[3] for word_tag in sent] #+++HANDE
            if self.config.model != 'LSTM' and self.config.model != 'BiLSTM':
                sents.append(["[CLS]"] + words + ["[SEP]"])
                tags_li.append(["<pad>"] + tags + ["<pad>"])
                values_li.append(["<pad>"] + values + ["<pad>"])
            else:
                sents.append(words)
                tags_li.append(tags)
                values_li.append(values)

        self.sents, self.tags_li, self.values_li = sents, tags_li, values_li
        if self.config.model == 'BertUncased':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

        self.tag_to_index = tag_to_index
        self.word_to_embid = word_to_embid

    def __len__(self):
        return len(self.sents)

    def convert_tokens_to_emb_ids(self, tokens):
        UNK_id = self.word_to_embid.get('UNK')
        return [self.word_to_embid.get(token, UNK_id) for token in tokens]

    def __getitem__(self, id):
        words, tags, values_li = self.sents[id], self.tags_li[id], self.values_li[id] # words, tags, values: string list

        x, y, values = [], [], [] # list of ids
        is_main_piece = [] # only score the main piece of each word
        for w, t, v in zip(words, tags, values_li):
            if self.config.model in ['LSTM', 'BiLSTM', 'LSTMRegression']:
                tokens = [w]
                xx = self.convert_tokens_to_emb_ids(tokens)
            else:
                tokens = self.tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
                xx = self.tokenizer.convert_tokens_to_ids(tokens)

            t = [t] + ["<pad>"] * (len(tokens) - 1)  # <PAD>: no decision
            yy = [self.tag_to_index[each] for each in t]  # (T,)

            head = [1] + [0]*(len(tokens) - 1) # identify the main piece of each word
            
            x.extend(xx)
            is_main_piece.extend(head)
            y.extend(yy)

        assert len(x) == len(y) == len(is_main_piece), "len(x)={}, len(y)={}, len(is_main_piece)={}".format(len(x), len(y), len(is_main_piece))
        # seqlen
        seqlen = len(y)

        # to string
        words = " ".join(words)
        tags = " ".join(tags)

        # Use log-values to remove affects of 0-skewed value distribution
        values = [np.log(float(v) + 1) if v not in ['<pad>','NA'] else self.config.invalid_set_to for v in values_li]

        return words, x, is_main_piece, tags, y, seqlen, values, self.config.invalid_set_to


def load_dataset(config):
    splits = dict()
    words = []
    all_sents = []
    for split in ['train', 'dev', 'test']:
        tagged_sents = []
        filename = config.train_set if split == 'train' else split
        with open(config.datadir+'/'+filename+'.txt') as f:
            lines = f.readlines()
            if config.fraction_of_train_data < 1 and split == 'train':
                slice = len(lines) * config.fraction_of_train_data
                lines = lines[0:int(round(slice))]
            sent = []
            for i, line in enumerate(lines):
                if line != "\n":
                    split_line = line.split('\t')
                    sent.append((split_line[0], split_line[1], split_line[2], split_line[3], split_line[4]))
                    words.append(split_line[0])
                elif line == "\n" or i+1 == len(lines):
                    tagged_sents.append(sent)
                    sent = []

        splits[split] = tagged_sents
        all_sents = all_sents + tagged_sents

    vocab = []
    for token in words:
        if token not in vocab:
            vocab.append(token)
    vocab = set(vocab)
    tags = list(set(word_tag[1] for sent in all_sents for word_tag in sent))
    tags = ["<pad>"] + tags

    tag_to_index = {tag: index for index, tag in enumerate(tags)}
    index_to_tag = {index: tag for index, tag in enumerate(tags)}

    print('Training sentences: {}'.format(len(splits["train"])))
    print('Dev sentences: {}'.format(len(splits["dev"])))
    print('Test sentences: {}'.format(len(splits["test"])))

    if config.sorted_batches:
        random.shuffle(splits["train"])
        splits["train"].sort(key=len)

    return splits, tag_to_index, index_to_tag, vocab


def pad(batch):
    # Pad sentences to the longest sample
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_main_piece = f(2)
    tags = f(3)
    seqlens = f(5)
    maxlen = np.array(seqlens).max()
    invalid_set_to = f(7)[0]

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = f(1, maxlen)
    y = f(4, maxlen)

    f = lambda x, seqlen: [sample[x] + [invalid_set_to] * (seqlen - len(sample[x])) for sample in batch] #invalid values are NA and <pad>
    values = f(6, maxlen)

    f = torch.LongTensor
    return words, f(x), is_main_piece, tags, f(y), seqlens, torch.FloatTensor(values), invalid_set_to


def load_embeddings(config, vocab):
    vocab.add('UNK') #FIXME: Do I need to "augmente" vocab permanently with UNK? I think both ways is ok. Correct?
    word2id = {word: id for id, word in enumerate(vocab)}
    embed_size = 300
    vocab_size = len(vocab)
    sd = 1 / np.sqrt(embed_size)
    weights = np.random.normal(0, scale=sd, size=[vocab_size, embed_size])
    weights = weights.astype(np.float32)
    with open(config.embedding_file, encoding='utf8', mode="r") as textFile:
        for line in textFile:
            line = line.split()
            word = line[0]

            # If word is in our vocab, then update the corresponding weights
            id = word2id.get(word, None)
            if id is not None and len(line) == 301:
                weights[id] = np.array([float(val) for val in line[1:]])

    return weights, word2id
