import numpy as np
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
parser.add_argument('--save_path',
                    type=str,
                    default='results.txt')
parser.add_argument('--log_every',
                    type=int,
                    default=10)
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.0005)
parser.add_argument('--gpu',
                    type=int,
                    default=0)


def main():

    config = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(config.gpu)
        device = torch.device('cuda:{}'.format(config.gpu))
        print("Training on GPU[{}]".format(config.gpu))
    else:
        print("GPU not available. Training on CPU.")
        device = 'cpu'

    train_data, test_data, tag_to_index, index_to_tag = prosody_dataset.load_data()

    model = Net(device, vocab_size=len(tag_to_index))
    model.to(device)
    model = nn.DataParallel(model)

    train_dataset = ProsodyDataset(train_data, tag_to_index)
    eval_dataset = ProsodyDataset(test_data, tag_to_index)

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 num_workers=1,
                                 collate_fn=prosody_dataset.pad)
    test_iter = data.DataLoader(dataset=eval_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=1,
                                collate_fn=prosody_dataset.pad)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Print the model
    print('Model:\n')
    print(model)
    print('\n')
    params = sum([p.numel() for p in model.parameters()])
    print('Parameters: {}'.format(params))

    train(model, train_iter, optimizer, criterion, config)
    evaluate(model, test_iter, tag_to_index, index_to_tag, config)


def train(model, iterator, optimizer, criterion, config):
    print('\nTraining started...\n')
    model.train()
    for i, batch in enumerate(iterator):
        words, x, tags, y, seqlens = batch
        optimizer.zero_grad()
        logits, y, _ = model(x, y) # logits: (N, T, VOCAB), y: (N, T)

        logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
        y = y.view(-1)  # (N*T,)

        loss = criterion(logits, y)
        loss.backward()

        optimizer.step()

        if i % config.log_every == 0:
            print("step: {}, loss: {}".format(i, loss.item()))


def evaluate(model, iterator, tag_to_index, index_to_tag, config):
    print('\nEvaluation started...\n')

    model.eval()

    Words, Tags, Y, Y_hat = [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, tags, y, seqlens = batch

            _, _, y_hat = model(x, y)  # y_hat: (N, T)

            Words.extend(words)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    # gets results and save
    with open(config.results, 'w') as results:
        for words, tags, y_hat in zip(Words, Tags, Y_hat):
            preds = [index_to_tag[hat] for hat in y_hat]
            # assert len(preds) == len(words.split()) == len(tags.split())
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds):
                results.write("{} {} {}\n".format(w, t, p))
            results.write("\n")

    # calc metric
    y_true = np.array([tag_to_index[line.split()[1]] for line in open(config.results, 'r').read().splitlines() if len(line) > 0])
    y_pred = np.array([tag_to_index[line.split()[2]] for line in open(config.results, 'r').read().splitlines() if len(line) > 0])

    acc = (y_true == y_pred).astype(np.int32).sum() / len(y_true)

    print("acc=%.2f\n" % acc)


if __name__ == "__main__":
    main()
