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
        self.bidirectional = True if config.model == 'BiLSTM' else False
        self.device = device
        self.linear = nn.Linear(1, config.embed_dim)
        self.fc = nn.Linear(config.hidden_dim*2, labels)
        self.word_embedding = nn.Embedding(vocab_size, config.embed_dim)
        self.lstm = nn.LSTM(input_size=config.embed_dim,
                           hidden_size=config.hidden_dim,
                           num_layers=config.layers,
                           dropout=0,
                           bidirectional=self.bidirectional)


    def forward(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        input = x.type(torch.FloatTensor).to(self.device)
        input = input.unsqueeze(2).permute(1, 0, 2)
        input = self.linear(input)
        enc = self.lstm(input)[0]
        enc = enc.permute(1, 0, 2)
        logits = self.fc(enc)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat
