import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
import prosody_dataset
from prosody_dataset import ProsodyDataset
from model import Net
from argparse import ArgumentParser

parser = ArgumentParser(description='Prosody prediction')

parser.add_argument('--batch_size',
                    type=int,
                    default=16)
parser.add_argument('--epochs',
                    type=int,
                    default=5)
parser.add_argument('--save_path',
                    type=str,
                    default='results.txt')
parser.add_argument('--log_every',
                    type=int,
                    default=50)
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.0001)
parser.add_argument('--weight_decay',
                    type=float,
                    default=0)
parser.add_argument('--gpu',
                    type=int,
                    default=0)
parser.add_argument('--number_of_sents',
                    type=int,
                    default=1000)
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


def main():

    config = parser.parse_args()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(config.gpu)
        device = torch.device('cuda:{}'.format(config.gpu))
        torch.cuda.manual_seed(config.seed)
        print("Training on GPU[{}]".format(config.gpu))
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

    train_data, test_data, dev_data, tag_to_index, index_to_tag = prosody_dataset.load_data(config)

    model = Net(device, vocab_size=len(tag_to_index))
    model.to(device)
    model = nn.DataParallel(model)

    train_dataset = ProsodyDataset(train_data, tag_to_index)
    eval_dataset = ProsodyDataset(dev_data, tag_to_index)
    test_dataset = ProsodyDataset(test_data, tag_to_index)

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

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    params = sum([p.numel() for p in model.parameters()])
    print('Parameters: {}'.format(params))

    print('\nTraining started...\n')
    for epoch in range(config.epochs):
        print("Epoch: {}".format(epoch+1))
        train(model, train_iter, optimizer, criterion, config)
        valid(model, dev_iter, tag_to_index, index_to_tag)

    test(model, test_iter, tag_to_index, index_to_tag, config)


def train(model, iterator, optimizer, criterion, config):
    model.train()
    for i, batch in enumerate(iterator):
        words, x, is_main_piece, tags, y, seqlens = batch
        optimizer.zero_grad()
        logits, y, _ = model(x, y) # logits: (N, T, VOCAB), y: (N, T)

        logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
        y = y.view(-1)  # (N*T,)

        loss = criterion(logits, y)
        loss.backward()

        optimizer.step()

        if i % config.log_every == 0 or i+1 == len(iterator):
            print("Training step: {}/{}, loss: {:<.4f}".format(i+1, len(iterator), loss.item()))


def valid(model, iterator, tag_to_index, index_to_tag):

    model.eval()

    Words, Is_main_piece, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_main_piece, tags, y, seqlens = batch

            _, _, y_hat = model(x, y)  # y_hat: (N, T)

            Words.extend(words)
            Is_main_piece.extend(is_main_piece)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    true = []
    predictions = []
    for words, is_main_piece, tags, y_hat in zip(Words, Is_main_piece, Tags, Y_hat):
        y_hat = [hat for head, hat in zip(is_main_piece, y_hat) if head == 1]
        preds = [index_to_tag[hat] for hat in y_hat]
        assert len(preds) == len(words.split()) == len(tags.split())
        for t, p in zip(tags.split()[1:-1], preds[1:-1]):
            true.append(tag_to_index[t])
            predictions.append(tag_to_index[p])

    # calc metric
    y_true = np.array(true)
    y_pred = np.array(predictions)
    acc = 100. * (y_true == y_pred).astype(np.int32).sum() / len(y_true)

    print('Validation accuracy: {:<5.2f}%\n'.format(round(acc, 2)))


def test(model, iterator, tag_to_index, index_to_tag, config):
    print('Calculating test accuracy and printing predictions to file {}'.format(config.save_path))
    print("Output file structure: <word>\t <tag>\t <prediction>\n")

    model.eval()

    Words, Is_main_piece, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_main_piece, tags, y, seqlens = batch

            _, _, y_hat = model(x, y)  # y_hat: (N, T)

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
    print('Test accuracy: {:<5.2f}% after {} epochs.\n'.format(round(acc, 2), config.epochs))


if __name__ == "__main__":
    main()
