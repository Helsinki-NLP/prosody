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
from model import Bert, LSTM, RegressionModel, WordMajority, ClassEncodings, BertAllLayers
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, f1_score, classification_report


parser = ArgumentParser(description='Prosody prediction')

parser.add_argument('--datadir',
                    type=str,
                    default='./data')
parser.add_argument('--train_set',
                    type=str,
                    choices=['train_100',
                             'train_360'],
                    default='BertUncased')
parser.add_argument('--batch_size',
                    type=int,
                    default=16)
parser.add_argument('--epochs',
                    type=int,
                    default=2)
parser.add_argument('--model',
                    type=str,
                    choices=['BertUncased',
                             'BertCased',
                             'LSTM',
                             'BiLSTM',
                             'Regression',
                             'WordMajority',
                             'ClassEncodings',
                             'BertAllLayers'],
                    default='BertUncased')
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
                    default=0.00005)
parser.add_argument('--weight_decay',
                    type=float,
                    default=0)
parser.add_argument('--gpu',
                    type=int,
                    default=None)
parser.add_argument('--fraction_of_sentences',
                    type=float,
                    default=1
                    )
parser.add_argument('--fraction_of_train_sentences',
                    type=float,
                    default=1
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
    elif config.model == "LSTM" or config.model == "BiLSTM":
        model = LSTM(device, config, vocab_size=len(vocab), labels=len(tag_to_index))
    elif config.model == "Regression":
        model = RegressionModel(device, config)
    elif config.model == "WordMajority":
        model = WordMajority(device, config, index_to_tag)
    elif config.model == "ClassEncodings":
        model = ClassEncodings(device, config, index_to_tag, tag_to_index)
    elif config.model == "BertAllLayers":
        model = BertAllLayers(device, config, labels=len(tag_to_index))
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

    if config.model in ["WordMajority"]:
        optimizer = None
    else:
        optimizer = optim_algorithm(model.parameters(),
                                    lr=config.learning_rate,
                                    weight_decay=config.weight_decay)

    if config.model == 'Regression':
        criterion = nn.MSELoss()
    elif config.model == 'ClassEncodings':
        criterion = nn.BCELoss()
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=0)

    params = sum([p.numel() for p in model.parameters()])
    print('Parameters: {}'.format(params))

    # TODO: implement word embeddings
    if config.model == 'LSTM' or config.model == 'BiLSTM':
        weights = load_embeddings(config, vocab)
        model.word_embedding.weight.data = torch.Tensor(weights).to(device)

    config.cells = config.layers

    if config.model == 'BiLSTM':
        config.cells *= 2

    if config.model == 'WordMajority': # 1 pass over the dataset is enough to collect stats
        config.epochs = 1

    print('\nTraining started...\n')
    for epoch in range(config.epochs):
        print("Epoch: {}".format(epoch+1))
        train(model, train_iter, optimizer, criterion, device, config)
        valid(model, dev_iter, criterion, tag_to_index, index_to_tag, device, config)

    test(model, test_iter, criterion, tag_to_index, index_to_tag, device, config)


def train(model, iterator, optimizer, criterion, device, config):
    if config.model == 'WordMajority' and model.load_stats():
        return

    model.train()
    for i, batch in enumerate(iterator):
        words, x, is_main_piece, tags, y, seqlens = batch

        if config.model == 'WordMajority':
            model.collect_stats(x, y)
            continue

        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        logits, y, _ = model(x, y) # logits: (N, T, VOCAB), y: (N, T)

        if config.model == 'Regression':
            loss = criterion(logits.to(device), y.float().to(device))
        elif config.model == 'ClassEncodings':
            logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
            y = y.view(-1, y.shape[-1])  # also (N*T, VOCAB)
            loss = criterion(logits.to(device), y.to(device))
        else:
            logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
            y = y.view(-1)  # (N*T,)
            loss = criterion(logits.to(device), y.to(device))

        loss.backward()

        optimizer.step()

        if i % config.log_every == 0 or i+1 == len(iterator):
            print("Training step: {}/{}, loss: {:<.4f}".format(i+1, len(iterator), loss.item()))

    if config.model == 'WordMajority':
        model.save_stats()


def valid(model, iterator, criterion, tag_to_index, index_to_tag, device, config):
    if config.model == 'WordMajority':
        return

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
                loss = criterion(logits.to(device), labels.float().to(device))
            elif config.model == 'ClassEncodings':
                logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
                labels = labels.view(-1, labels.shape[-1])  # also (N*T, VOCAB)
                loss = criterion(logits.to(device), labels.to(device))
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
        if config.model != 'LSTM' and config.model != 'BiLSTM':
            tagslice = tags.split()[1:-1]
            predsslice = preds[1:-1]
            assert len(preds) == len(words.split()) == len(tags.split())
        else:
            tagslice = tags.split()
            predsslice = preds
        for t, p in zip(tagslice, predsslice):
            if config.ignore_punctuation:
                if t != 'NA':
                    true.append(t)
                    predictions.append(p)
            else:
                true.append(t)
                predictions.append(p)

    # calc metric

    y_true = np.array(true)
    y_pred = np.array(predictions)
    # acc = 100. * (y_true == y_pred).astype(np.int32).sum() / len(y_true)

    #true_labels = np.array([index_to_tag[index] for index in true])
    #predicted_labels = np.array([index_to_tag[index] for index in predictions])

    accuracy = accuracy_score(y_true, y_pred)

    print('Validation accuracy: {:<5.2f}%, Validation loss: {:<.4f}\n'.format(round(100. * accuracy, 2), np.mean(dev_losses)))


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

            if config.model in ['Regression']:
                loss = criterion(logits.to(device), labels.float().to(device))
            elif config.model == 'ClassEncodings':
                logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
                labels = labels.view(-1, labels.shape[-1])  # also (N*T, VOCAB)
                loss = criterion(logits.to(device), labels.to(device))
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
            if config.model != 'LSTM' and config.model != 'BiLSTM':
                tagslice = tags.split()[1:-1]
                predsslice = preds[1:-1]
                wordslice = words.split()[1:-1]
                assert len(preds) == len(words.split()) == len(tags.split())
            else:
                tagslice = tags.split()
                predsslice = preds
                wordslice = words.split()
            for w, t, p in zip(wordslice, tagslice, predsslice):
                results.write("{}\t{}\t{}\n".format(w, t, p))
                if config.ignore_punctuation:
                    if t != 'NA':
                        true.append(t)
                        predictions.append(p)
                else:
                    true.append(t)
                    predictions.append(p)
            results.write("\n")

    # calc metric
    y_true = np.array(true)
    y_pred = np.array(predictions)

    #y_true = np.array([index_to_tag[index] for index in true])
    #y_pred = np.array([index_to_tag[index] for index in predictions])

    #classes = ['0', '1', '2'] if config.ignore_punctuation else ['0', '1', '2', 'NA']
    classes = ['0', '1', '2', 'NA']


    np.set_printoptions(precision=1)
    plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix - ' + config.model)

    plot_name = 'images/confusion_matrix-'+ config.model+'.png' if config.ignore_punctuation else 'confusion_matrix-'+ config.model+'no_NA.png'
    plt.savefig(plot_name)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')


    print('\nAccuracy: {}'.format(round(accuracy, 4)))
    print('F1 score: {}'.format(round(f1, 4)))
    print('Recall: {}'.format(round(recall, 4)))
    print('Precision: {}'.format(round(precision, 4)))

    # target_names = ['label 0', 'label 1', 'label 2'] if config.ignore_punctuation else ['label 0', 'label 1', 'label 2', 'label NA']
    # target_names = ['label 0', 'label 1', 'label 2']
    # print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    # acc = 100. * (y_true == y_pred).astype(np.int32).sum() / len(y_true)
    # print('Test accuracy: {:<5.2f}%, Test loss: {:<.4f} after {} epochs.\n'.format(round(acc, 2), np.mean(test_losses), config.epochs))


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if not title:
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
    else:
        print(title)

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


if __name__ == "__main__":
    main()
