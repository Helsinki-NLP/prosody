import torch
from torch import nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel


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
