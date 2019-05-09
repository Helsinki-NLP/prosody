import os
from torch.utils import data
from pytorch_pretrained_bert import BertTokenizer
from sklearn.model_selection import train_test_split
import nltk
import torch
import numpy as np

DATADIR = "data/"


class ProsodyDataset(data.Dataset):
    def __init__(self, tagged_sents, tag2id):
        sents, tags_li = [], [] # list of lists
        for sent in tagged_sents:
            words = [word_tag[0] for word_tag in sent]
            tags = [word_tag[1] for word_tag in sent]
            sents.append(["[CLS]"] + words + ["[SEP]"])
            tags_li.append(["<pad>"] + tags + ["<pad>"])
        self.sents, self.tags_li = sents, tags_li
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
        self.tag2id = tag2id

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, id):
        words, tags = self.sents[id], self.tags_li[id] # words, tags: string list

        # We give credits only to the first piece.
        x, y = [], [] # list of ids
        is_heads = [] # list. 1: the token is the first piece of a word
        for w, t in zip(words, tags):
            tokens = self.tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
            xx = self.tokenizer.convert_tokens_to_ids(tokens)

            is_head = [1] + [0]*(len(tokens) - 1)

            t = [t] + ["<pad>"] * (len(tokens) - 1)  # <PAD>: no decision
            yy = [self.tag2id[each] for each in t]  # (T,)

            x.extend(xx)
            is_heads.extend(is_head)
            y.extend(yy)

        assert len(x)==len(y)==len(is_heads), "len(x)={}, len(y)={}, len(is_heads)={}".format(len(x), len(y), len(is_heads))

        # seqlen
        seqlen = len(y)

        # to string
        words = " ".join(words)
        tags = " ".join(tags)
        return words, x, is_heads, tags, y, seqlen


def load_pos_data():
    tagged_sents = nltk.corpus.treebank.tagged_sents()
    tags = list(set(word_tag[1] for sent in tagged_sents for word_tag in sent))
    tags = ["<pad>"] + tags

    tag2id = {tag: id for id, tag in enumerate(tags)}
    id2tag = {id: tag for id, tag in enumerate(tags)}

    # Let's split the data into train and test (or eval)
    train_data, test_data = train_test_split(tagged_sents, test_size=.1)
    len(train_data), len(test_data)

    return train_data, test_data, tag2id, id2tag


def load_data():
    # TODO: update with prosody dataset and loading
    # TODO: Load sentences into a list as lists of tuples [[('word', 'tag), ...], ...]
    directory = os.fsencode(DATADIR)
    tagged_sents = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            with open(DATADIR+filename) as f:
                lines = f.read().splitlines()
                sent = []
                for line in lines:
                    split_line = line.split('\t')
                    sent.append((split_line[0], split_line[1]))
            tagged_sents.append(sent)
            continue
        else:
            break

    tags = list(set(word_tag[1] for sent in tagged_sents for word_tag in sent))
    tags = ["<pad>"] + tags

    tag_to_index = {tag: id for id, tag in enumerate(tags)}
    index_to_tag = {id: tag for id, tag in enumerate(tags)}

    # Let's split the data into train and test (or eval)
    train_data, test_data = train_test_split(tagged_sents, test_size=.1)
    len(train_data), len(test_data)

    return train_data, test_data, tag_to_index, index_to_tag


def pad(batch):
    # Pads to the longest sample
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-1)
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: <pad>
    x = f(1, maxlen)
    y = f(-2, maxlen)

    f = torch.LongTensor
    return words, f(x), is_heads, tags, f(y), seqlens
