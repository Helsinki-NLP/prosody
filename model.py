import sys
import os
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
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


class BertLSTM(nn.Module):
    def __init__(self, device, config, labels=None):
        super().__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.fc = nn.Linear(1536, labels).to(device)
        self.device = device
        self.lstm = nn.LSTM(input_size=768,
                            hidden_size=768,
                            num_layers=1,
                            dropout=0,
                            bidirectional=True)

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

        enc = enc.permute(1, 0, 2)
        enc = self.lstm(enc)[0]
        enc = enc.permute(1, 0, 2)
        logits = self.fc(enc).to(self.device)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat


class LSTM(nn.Module):
    def __init__(self, device, config, vocab_size, labels=None):
        super().__init__()
        self.config = config
        self.bidirectional = True if config.model == 'BiLSTM' else False
        self.device = device
        self.dropout = 0 if config.layers == 1 else 0.2
        hidden_dim = config.hidden_dim*2 if config.model == 'BiLSTM' else config.hidden_dim
        self.fc = nn.Linear(hidden_dim, labels)
        self.word_embedding = nn.Embedding(vocab_size, 300)
        self.lstm = nn.LSTM(input_size=300,
                           hidden_size=config.hidden_dim,
                           num_layers=config.layers,
                           dropout=self.dropout,
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
    def __init__(self, device, config, index_to_tag):
        super().__init__()
        self.device = device
        self.config = config
        self.index_to_tag = index_to_tag
        self.nr_classes = len(index_to_tag)
        self.majorityClass = dict()
        self.stats_file = self.config.datadir + '/train.classes.json'
        self.valid_classes = [1,2,3]

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

            if int(class_idx) not in self.valid_classes: # a couple of pads come along here. why? data problem?
                continue

            if word_idx not in self.majorityClass.keys():
                self.majorityClass[word_idx] = {str(cls): 0 for cls in self.valid_classes}

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
                preds.append(1)

        logits[np.arange(x.shape[0]*x.shape[1]), preds] = 1
        logits = logits.view(x.shape[0], x.shape[1], self.nr_classes).to(self.device)

        y_hat = logits.argmax(-1)
        return logits, y, y_hat


class ClassEncodings(nn.Module):
    def __init__(self, device, config, index_to_tag, tag_to_index):
        super().__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.FIXED_NR_OUTPUTS = 8 # FIXME: We may want to make this dynamic?
                                  # For now it is handcoded to the mapping below.

        self.fc = nn.Linear(768, self.FIXED_NR_OUTPUTS).to(device)

        self.device = device

        self.index_to_tag = index_to_tag
        self.tag_to_index = tag_to_index

        self.mapping = {'<pad>': [0, 0, 0, 0, 0, 0, 1, 1],  # <pad>
                        'NA'   : [0, 0, 0, 0, 1, 1, 0, 0],  # NA
                        '2'    : [0, 0, 1, 1, 0, 0, 0, 0],  # prosody value 2
                        '0'    : [1, 1, 0, 0, 0, 0, 0, 0],  # prosody value 0
                        '1'    : [0, 1, 1, 0, 0, 0, 0, 0]}  # prosody value 1


    def get_encoding(self, index):
        return self.mapping[self.index_to_tag[index]]

    def get_tag(self, encoding):
        distance = lambda L1,L2: sum([abs(L1[i]-L2[i]) for i in range(len(L1))])
        distances_to_classes = {tag:distance(self.mapping[tag], encoding) for tag in self.mapping.keys()}
        return min(distances_to_classes, key=distances_to_classes.get)

    def forward(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)

        batch_size = x.shape[0]
        seq_length = x.shape[1]

        if self.training:
            self.bert.train()
            encoded_layers, _ = self.bert(x)
            enc = encoded_layers[-1]
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers, _ = self.bert(x)
                enc = encoded_layers[-1]

        logits = F.sigmoid(self.fc(enc)).to(self.device)

        y_hat = torch.LongTensor([self.tag_to_index[self.get_tag(logit)]\
                                                                for logit in logits.view(batch_size * seq_length, -1)])\
                                                                .view(batch_size, seq_length).to(self.device)

        class_encodings = torch.FloatTensor([self.get_encoding(label.item()) for label in y.view(-1)])
        class_encodings = class_encodings.view(batch_size, seq_length, self.FIXED_NR_OUTPUTS).to(self.device)

        return logits, class_encodings, y_hat



class BertAllLayers(nn.Module):
    def __init__(self, device, config, labels=None):
        super().__init__()

        if config.model == "BertCased":
            self.bert = BertModel.from_pretrained('bert-base-cased')
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.fc = nn.Linear(768*12, labels).to(device)
        self.device = device

    def forward(self, x, y):

        x = x.to(self.device)
        y = y.to(self.device)

        if self.training:
            self.bert.train()
            encoded_layers, _ = self.bert(x)
            enc = torch.cat([encoded_layers[i] for i in range(len(encoded_layers))], dim=2)
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers, _ = self.bert(x)
                enc = torch.cat([encoded_layers[i] for i in range(len(encoded_layers))], dim = 2)

        logits = self.fc(enc).to(self.device)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat
