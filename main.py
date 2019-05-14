import os
import sys
import errno
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
import prosody_dataset
from prosody_dataset import Dataset
from prosody_dataset import load_embeddings
from model import Bert, LSTM
from regression_model import RegressionModel
from argparse import ArgumentParser

parser = ArgumentParser(description='Prosody prediction')

parser.add_argument('--datadir',
                    type=str,
                    default='./data')
parser.add_argument('--batch_size',
                    type=int,
                    default=64)
parser.add_argument('--epochs',
                    type=int,
                    default=5)
parser.add_argument('--model',
                    type=str,
                    choices=['BertUncased',
                             'BertCased',
                             'LSTM',
                             'Regression'],
                    default='LSTM')
parser.add_argument('--no_brnn',
                    action='store_false',
                    dest='bidirectional')
parser.add_argument('--hidden_dim',
                    type=int,
                    default=600)
parser.add_argument('--embedding_file',
                    type=str,
                    default='embeddings/glove.840B.300d.txt')
parser.add_argument('--layers',
                    type=int,
                    default=1)
parser.add_argument('--save_path',
                    type=str,
                    default='results.txt')
parser.add_argument('--log_every',
                    type=int,
                    default=10)
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.001)
parser.add_argument('--weight_decay',
                    type=float,
                    default=0)
parser.add_argument('--gpu',
                    type=int,
                    default=None)
parser.add_argument('--fraction_of_sentences',
                    type=float,
                    default=0.02
                    )
parser.add_argument("--optimizer",
                    type=str,
                    choices=['rprop',
                             'adadelta',
                             'adagrad',
                             'rmsprop',
                             'adamax',
                             'asgd',
                             'adam',
                             'sgd'],
                    default='adam')
parser.add_argument('--include_punctuation',
                    action='store_false',
                    dest='ignore_punctuation')
parser.add_argument('--seed',
                    type=int,
                    default=1234)


def make_dirs(name):
    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


def main():

    config = parser.parse_args()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    if config.gpu is not None:
        torch.cuda.set_device(config.gpu)
        device = torch.device('cuda:{}'.format(config.gpu))
        torch.cuda.manual_seed(config.seed)
        print("\nTraining on GPU[{}] (torch.device({})).".format(config.gpu, device))
    else:
        device = torch.device('cpu')
        print("GPU not available so training on CPU (torch.device({})).".format(device))
        device = 'cpu'


    # Optimizer
    if config.optimizer == 'adadelta':
        optim_algorithm = optim.Adadelta
    elif config.optimizer == 'adagrad':
        optim_algorithm = optim.Adagrad
    elif config.optimizer == 'adam':
        optim_algorithm = optim.Adam
    elif config.optimizer == 'adamax':
        optim_algorithm = optim.Adamax
    elif config.optimizer == 'asgd':
        optim_algorithm = optim.ASGD
    elif config.optimizer == 'rmsprop':
        optim_algorithm = optim.RMSprop
    elif config.optimizer == 'rprop':
        optim_algorithm = optim.Rprop
    elif config.optimizer == 'sgd':
        optim_algorithm = optim.SGD
    else:
        raise Exception('Unknown optimization optimizer: "%s"' % config.optimizer)

    splits, tag_to_index, index_to_tag, vocab = prosody_dataset.load_dataset(config)

    if config.model == "BertUncased" or config.model == "BertCased":
        model = Bert(device, config, labels=len(tag_to_index))
    elif config.model == "LSTM":
        model = LSTM(device, config, vocab_size=len(vocab), labels=len(tag_to_index))
    elif config.model == "Regression":
        model = RegressionModel(device, config)
    else:
        raise NotImplementedError("Model option not supported.")

    model.to(device)

    train_dataset = Dataset(splits["train"], tag_to_index, config)
    eval_dataset = Dataset(splits["dev"], tag_to_index, config)
    test_dataset = Dataset(splits["test"], tag_to_index, config)

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 num_workers=1,
                                 collate_fn=prosody_dataset.pad)
    dev_iter = data.DataLoader(dataset=eval_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=1,
                                collate_fn=prosody_dataset.pad)
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=1,
                                collate_fn=prosody_dataset.pad)

    optimizer = optim_algorithm(model.parameters(),
                                lr=config.learning_rate,
                                weight_decay=config.weight_decay)

    if config.model == 'Regression':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=0)

    params = sum([p.numel() for p in model.parameters()])
    print('Parameters: {}'.format(params))

    # TODO: implement word embeddings
    if config.model == 'LSTM':
        weights = load_embeddings(config, vocab)
        model.word_embedding.weight.data = torch.Tensor(weights).to(device)

    config.cells = config.layers

    if config.bidirectional:
        config.cells *= 2

    print('\nTraining started...\n')
    for epoch in range(config.epochs):
        print("Epoch: {}".format(epoch+1))
        train(model, train_iter, optimizer, criterion, device, config)
        valid(model, dev_iter, criterion, tag_to_index, index_to_tag, device, config)

    test(model, test_iter, criterion, tag_to_index, index_to_tag, device, config)


def train(model, iterator, optimizer, criterion, device, config):
    model.train()
    for i, batch in enumerate(iterator):
        words, x, is_main_piece, tags, y, seqlens = batch
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        logits, y, _ = model(x, y) # logits: (N, T, VOCAB), y: (N, T)

        if config.model == 'Regression':
            loss = criterion(logits, y.float())
        else:
            logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
            y = y.view(-1)  # (N*T,)
            loss = criterion(logits.to(device), y.to(device))

        loss.backward()

        optimizer.step()

        if i % config.log_every == 0 or i+1 == len(iterator):
            print("Training step: {}/{}, loss: {:<.4f}".format(i+1, len(iterator), loss.item()))


def valid(model, iterator, criterion, tag_to_index, index_to_tag, device, config):

    model.eval()
    dev_losses = []
    Words, Is_main_piece, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_main_piece, tags, y, seqlens = batch
            x = x.to(device)
            y = y.to(device)
            logits, labels, y_hat = model(x, y)  # y_hat: (N, T)

            if config.model == 'Regression':
                loss = criterion(logits, labels.float())
            else:
                logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
                labels = labels.view(-1)  # (N*T,)
                loss = criterion(logits.to(device), labels.to(device))

            dev_losses.append(loss.item())

            Words.extend(words)
            Is_main_piece.extend(is_main_piece)
            Tags.extend(tags)
            Y.extend(y.cpu().numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    true = []
    predictions = []
    for words, is_main_piece, tags, y_hat in zip(Words, Is_main_piece, Tags, Y_hat):
        y_hat = [hat for head, hat in zip(is_main_piece, y_hat) if head == 1]
        if config.model=='Regression':
            preds = [index_to_tag[max(min(int(hat), 4), 0)] for hat in y_hat]
        else:
            preds = [index_to_tag[hat] for hat in y_hat]
        if config.model != 'LSTM':
            tagslice = tags.split()[1:-1]
            predsslice = preds[1:-1]
            assert len(preds) == len(words.split()) == len(tags.split())
        else:
            tagslice = tags.split()
            predsslice = preds
        for t, p in zip(tagslice, predsslice):
            true.append(tag_to_index[t])
            predictions.append(tag_to_index[p])

    # calc metric
    y_true = np.array(true)
    y_pred = np.array(predictions)
    acc = 100. * (y_true == y_pred).astype(np.int32).sum() / len(y_true)

    print('Validation accuracy: {:<5.2f}%, Validation loss: {:<.4f}\n'.format(round(acc, 2), np.mean(dev_losses)))


def test(model, iterator, criterion, tag_to_index, index_to_tag, device, config):
    print('Calculating test accuracy and printing predictions to file {}'.format(config.save_path))
    print("Output file structure: <word>\t <tag>\t <prediction>\n")

    model.eval()
    test_losses = []

    Words, Is_main_piece, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_main_piece, tags, y, seqlens = batch
            x = x.to(device)
            y = y.to(device)
            logits, labels, y_hat = model(x, y)  # y_hat: (N, T)

            if config.model == 'Regression':
                loss = criterion(logits, labels.float())
            else:
                logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
                labels = labels.view(-1)  # (N*T,)
                loss = criterion(logits, labels)

            test_losses.append(loss.item())

            Words.extend(words)
            Is_main_piece.extend(is_main_piece)
            Tags.extend(tags)
            Y.extend(y.cpu().numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    true = []
    predictions = []
    # gets results and save
    with open(config.save_path, 'w') as results:
        for words, is_main_piece, tags, y_hat in zip(Words, Is_main_piece, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_main_piece, y_hat) if head == 1]
            if config.model == 'Regression':
                preds = [index_to_tag[max(min(int(hat), 4), 0)] for hat in y_hat]
            else:
                preds = [index_to_tag[hat] for hat in y_hat]
            if config.model != 'LSTM':
                tagslice = tags.split()[1:-1]
                predsslice = preds[1:-1]
                wordslice = words.split()[1:-1]
                assert len(preds) == len(words.split()) == len(tags.split())
            else:
                tagslice = tags.split()
                predsslice = preds
                wordslice = words.split()
            for w, t, p in zip(wordslice, tagslice, predsslice):
                results.write("{} {} {}\n".format(w, t, p))
                true.append(tag_to_index[t])
                predictions.append(tag_to_index[p])
            results.write("\n")

    # calc metric
    y_true = np.array(true)
    y_pred = np.array(predictions)
    acc = 100. * (y_true == y_pred).astype(np.int32).sum() / len(y_true)
    print('Test accuracy: {:<5.2f}%, Test loss: {:<.4f} after {} epochs.\n'.format(round(acc, 2), np.mean(test_losses), config.epochs))


if __name__ == "__main__":
    main()
