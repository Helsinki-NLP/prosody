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
                           dropout=0,
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
