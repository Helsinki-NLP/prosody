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
from model import Bert, LSTM
from regression_model import RegressionModel
from argparse import ArgumentParser

parser = ArgumentParser(description='Prosody prediction')

parser.add_argument('--datadir',
                    type=str,
                    default='./data')
parser.add_argument('--batch_size',
                    type=int,
                    default=16)
parser.add_argument('--epochs',
                    type=int,
                    default=3)
parser.add_argument('--model',
                    type=str,
                    choices=['BertUncased',
                             'BertCased',
                             'LSTM',
                             'BiLSTM',
			     'Regression'],
                    default='BertUncased')
parser.add_argument('--embed_dim',
                    type=int,
                    default=300)
parser.add_argument('--hidden_dim',
                    type=int,
                    default=1200)
parser.add_argument('--word_embedding',
                    type=str,
                    default=None)
parser.add_argument('--layers',
                    type=int,
                    default=3)
parser.add_argument('--save_path',
                    type=str,
                    default='results.txt')
parser.add_argument('--log_every',
                    type=int,
                    default=10)
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.00005)
parser.add_argument('--weight_decay',
                    type=float,
                    default=0)
parser.add_argument('--gpu',
                    type=int,
                    default=0)
parser.add_argument('--number_of_sents',
                    type=int,
                    default=500)
parser.add_argument('--test_and_dev_split',
                    type=float,
                    default=.1)
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

    if torch.cuda.is_available():
        torch.cuda.set_device(config.gpu)
        device = torch.device('cuda:{}'.format(config.gpu))
        torch.cuda.manual_seed(config.seed)
        print("\nTraining on GPU[{}]".format(config.gpu))
    else:
        print("GPU not available. Training on CPU.")
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

    train_data, test_data, dev_data, tag_to_index, index_to_tag, vocab_size = prosody_dataset.load_dataset(config)

    if config.model == "BertUncased" or config.model == "BertCased":
        model = Bert(device, config, labels=len(tag_to_index))
    elif config.model == "LSTM" or config.model == "BiLSTM":
        model = LSTM(device, config, vocab_size=vocab_size, labels=len(tag_to_index))
    elif config.model == "Regression":
        model = RegressionModel(device, config)
    else:
        raise NotImplementedError("Only BERT models are supported at this moment. Use BertCased or BertUncased.")

    model.to(device)
    model = nn.DataParallel(model)

    train_dataset = Dataset(train_data, tag_to_index)
    eval_dataset = Dataset(dev_data, tag_to_index)
    test_dataset = Dataset(test_data, tag_to_index)

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

    if config.word_embedding:
        pretrained_embedding = os.path.join(os.getcwd(),
                                            '.vector_cache/' + config.corpus + '_' + config.word_embedding + '.pt')
        if os.path.isfile(pretrained_embedding):
            inputs.vocab.vectors = torch.load(pretrained_embedding, map_location=device)
        else:
            print('Downloading pretrained {} word embeddings\n'.format(config.word_embedding))
            inputs.vocab.load_vectors(config.word_embedding)
            make_dirs(os.path.dirname(pretrained_embedding))
            torch.save(inputs.vocab.vectors, pretrained_embedding)

    config.cells = config.layers

    if config.model == 'BiLSMT':
        config.cells *= 2

    print('\nTraining started...\n')
    for epoch in range(config.epochs):
        print("Epoch: {}".format(epoch+1))
        train(model, train_iter, optimizer, criterion, config)
        valid(model, dev_iter, criterion, tag_to_index, index_to_tag)

    test(model, test_iter, criterion, tag_to_index, index_to_tag, config)


def train(model, iterator, optimizer, criterion, config):
    model.train()
    for i, batch in enumerate(iterator):
        words, x, is_main_piece, tags, y, seqlens = batch
        optimizer.zero_grad()
        logits, y, _ = model(x, y) # logits: (N, T, VOCAB), y: (N, T)

        if config.model == 'Regression':
            loss = criterion(logits, y)
        else:
            logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
            y = y.view(-1)  # (N*T,)
            loss = criterion(logits, y)

        loss.backward()

        optimizer.step()

        if i % config.log_every == 0 or i+1 == len(iterator):
            print("Training step: {}/{}, loss: {:<.4f}".format(i+1, len(iterator), loss.item()))


def valid(model, iterator, criterion, tag_to_index, index_to_tag):

    model.eval()
    dev_losses = []
    Words, Is_main_piece, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_main_piece, tags, y, seqlens = batch

            logits, labels, y_hat = model(x, y)  # y_hat: (N, T)
            logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
            labels = labels.view(-1)  # (N*T,)

            loss = criterion(logits, labels)
            dev_losses.append(loss.item())

            Words.extend(words)
            Is_main_piece.extend(is_main_piece)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    true = []
    predictions = []
    for words, is_main_piece, tags, y_hat in zip(Words, Is_main_piece, Tags, Y_hat):
        y_hat = [hat for head, hat in zip(is_main_piece, y_hat) if head == 1]
        if config.model=='Regression':
            preds = [index_to_tag[max(min(int(hat), 4), 0)] for hat in y_hat]
        else:
            preds = [index_to_tag[hat] for hat in y_hat]
        assert len(preds) == len(words.split()) == len(tags.split())
        for t, p in zip(tags.split()[1:-1], preds[1:-1]):
            true.append(tag_to_index[t])
            predictions.append(tag_to_index[p])

    # calc metric
    y_true = np.array(true)
    y_pred = np.array(predictions)
    acc = 100. * (y_true == y_pred).astype(np.int32).sum() / len(y_true)

    print('Validation accuracy: {:<5.2f}%, Validation loss: {:<.4f}\n'.format(round(acc, 2), np.mean(dev_losses)))


def test(model, iterator, criterion, tag_to_index, index_to_tag, config):
    print('Calculating test accuracy and printing predictions to file {}'.format(config.save_path))
    print("Output file structure: <word>\t <tag>\t <prediction>\n")

    model.eval()
    test_losses = []

    Words, Is_main_piece, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_main_piece, tags, y, seqlens = batch

            logits, labels, y_hat = model(x, y)  # y_hat: (N, T)
            logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
            labels = labels.view(-1)  # (N*T,)

            loss = criterion(logits, labels)
            test_losses.append(loss.item())

            Words.extend(words)
            Is_main_piece.extend(is_main_piece)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    true = []
    predictions = []
    # gets results and save
    with open(config.save_path, 'w') as results:
        for words, is_main_piece, tags, y_hat in zip(Words, Is_main_piece, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_main_piece, y_hat) if head == 1]
            if config.model=='Regression':
                preds = [index_to_tag[max(min(int(hat), 4), 0)] for hat in y_hat]
            else:
                preds = [index_to_tag[hat] for hat in y_hat]
            assert len(preds) == len(words.split()) == len(tags.split())
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
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
