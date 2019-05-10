import torch
from torch import nn
from pytorch_pretrained_bert import BertModel


class Bert(nn.Module):
    def __init__(self, device, config, vocab_size=None):
        super().__init__()

        if config.model == "BertCased":
            self.bert = BertModel.from_pretrained('bert-base-cased')
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.fc = nn.Linear(768, vocab_size)
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

        logits = self.fc(enc)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat


class LSTM(nn.Module):
    def __init__(self, device, config, vocab_size=None):
        super().__init__()
        self.config = config
        self.bidirectional = True if config.model == 'BiLSTM' else False
        self.fc = nn.Linear(600, vocab_size)
        self.word_embedding = nn.Embedding(vocab_size, config.embed_dim)
        self.lstm = nn.LSTM(input_size=config.embed_dim,
                           hidden_size=config.hidden_dim,
                           num_layers=config.layers,
                           dropout=0,
                           bidirectional=self.bidirectional)


    def forward(self, x, y):
        #x = self.word_embedding(x)
        h_0 = c_0 = torch.Tensor(x.data.new(self.config.cells,
                                            self.config.batch_size,
                                             self.config.hidden_dim).zero_())
        enc = self.lstm(x, (h_0, c_0))[1][0]

        logits = self.fc(enc)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat
