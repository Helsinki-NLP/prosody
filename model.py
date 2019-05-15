import sys
import os
import json
import numpy as np
import torch
from torch import nn
from pytorch_pretrained_bert import BertModel


class Bert(nn.Module):
    def __init__(self, device, config, labels=None):
        super().__init__()

        if config.model == "BertCased":
            self.bert = BertModel.from_pretrained('bert-base-cased')
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.fc = nn.Linear(768, labels).to(device)
        self.device = device

    def forward(self, x, y):

        x = x.to(self.device)
        y = y.to(self.device)

        if self.training:
            self.bert.train()
            encoded_layers, _ = self.bert(x)
            enc = encoded_layers[-1]
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers, _ = self.bert(x)
                enc = encoded_layers[-1]

        logits = self.fc(enc).to(self.device)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat


class LSTM(nn.Module):
    def __init__(self, device, config, vocab_size, labels=None):
        super().__init__()
        self.config = config
        self.bidirectional = config.bidirectional
        self.device = device
        hidden_dim = config.hidden_dim*2 if config.bidirectional else config.hidden_dim
        self.fc = nn.Linear(hidden_dim, labels)
        self.word_embedding = nn.Embedding(vocab_size, 300)
        self.lstm = nn.LSTM(input_size=300,
                           hidden_size=config.hidden_dim,
                           num_layers=config.layers,
                           dropout=0.2,
                           bidirectional=self.bidirectional)

    def forward(self, x, y):
        x = x.permute(1, 0).to(self.device)
        y = y.to(self.device)
        emb = self.word_embedding(x)
        enc = self.lstm(emb)[0]
        enc = enc.permute(1, 0, 2)
        logits = self.fc(enc)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat


class RegressionModel(nn.Module):
    def __init__(self, device, config):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.fc = nn.Linear(768, 1)
        self.device = device

    def forward(self, x, y):
        '''
        x: (N, T). int64
        y: (N, T). int64
        '''
        x = x.to(self.device)
        y = y.to(self.device)

        if self.training:
            self.bert.train()
            encoded_layers, _ = self.bert(x)
            enc = encoded_layers[-1]
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers, _ = self.bert(x)
                enc = encoded_layers[-1]

        out = self.fc(enc).squeeze()
        return out, y, out


class WordMajority(nn.Module):
    def __init__(self, device, config, tag_to_index):
        super().__init__()
        self.device = device
        self.config = config
        self.tag_to_index = tag_to_index
        self.nr_classes = len(tag_to_index)
        self.majorityClass = dict()
        self.stats_file = self.config.datadir + '/train.classes.json'

    def load_stats(self):
        if os.path.isfile(self.stats_file):
            self.majorityClass = json.load(open(self.stats_file))
            return True
        return False

    def save_stats(self):
        jsondict = json.dumps(self.majorityClass)
        with open(self.stats_file, 'w') as fout:
            fout.write(jsondict)

    def collect_stats(self, x, y):
        x_list = x.view(-1).tolist()
        y_list = y.view(-1).tolist()

        for idx in range(x.shape[0] * x.shape[1]):
            word_idx = str(x_list[idx])
            class_idx = str(y_list[idx])

            if word_idx not in self.majorityClass.keys():
                self.majorityClass[word_idx] = {str(cls): 0 for cls in range(self.nr_classes)}

            try:
                self.majorityClass[word_idx][class_idx] += 1
            except:
                print('Exception in WordMajority::collect_stats():')
                print('word_idx:', word_idx, 'class_idx:', class_idx)
                print('majorityClass keys are:', self.majorityClass.keys())
                if word_idx in self.majorityClass.keys():
                    print('majorityClass[word_idx] exists, keys are:', self.majorityClass[word_idx].keys())
                sys.exit(1)


    def forward(self, x, y):
        '''
        x: (N, T). int64
        y: (N, T). int64
        '''
        x = x.to(self.device)
        y = y.to(self.device)

        logits = torch.zeros(x.shape[0]*x.shape[1], self.nr_classes)

        preds = []
        for word_idx_tensor in x.view(-1):
            word_idx_str = str(word_idx_tensor.item())
            if word_idx_str in self.majorityClass.keys():
                preds.append(int(max(self.majorityClass[word_idx_str], key=self.majorityClass[word_idx_str].get)))
            else:
                preds.append(0)

        logits[np.arange(x.shape[0]*x.shape[1]), preds] = 1
        logits = logits.view(x.shape[0], x.shape[1], self.nr_classes).to(self.device)

        y_hat = logits.argmax(-1)
        return logits, y, y_hat
