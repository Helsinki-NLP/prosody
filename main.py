import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
import prosody_dataset
from prosody_dataset import ProsodyDataset
from model import Net


def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data, test_data, tag_to_index, index_to_tag = prosody_dataset.load_data()

    model = Net(device, vocab_size=len(tag_to_index))
    model.to(device)
    model = nn.DataParallel(model)

    train_dataset = ProsodyDataset(train_data, tag_to_index)
    eval_dataset = ProsodyDataset(test_data, tag_to_index)

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=8,
                                 shuffle=True,
                                 num_workers=1,
                                 collate_fn=prosody_dataset.pad)
    test_iter = data.DataLoader(dataset=eval_dataset,
                                batch_size=8,
                                shuffle=False,
                                num_workers=1,
                                collate_fn=prosody_dataset.pad)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    train(model, train_iter, optimizer, criterion)
    evaluate(model, test_iter, tag_to_index, index_to_tag)


def train(model, iterator, optimizer, criterion):
    model.train()
    for i, batch in enumerate(iterator):
        words, x, is_heads, tags, y, seqlens = batch
        _y = y # for monitoring
        optimizer.zero_grad()
        logits, y, _ = model(x, y) # logits: (N, T, VOCAB), y: (N, T)

        logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
        y = y.view(-1)  # (N*T,)

        loss = criterion(logits, y)
        loss.backward()

        optimizer.step()

        if i%100==0: # monitoring
            print("step: {}, loss: {}".format(i, loss.item()))


def evaluate(model, iterator, tag_to_index, index_to_tag):
    model.eval()

    words, is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch

            _, _, y_hat = model(x, y)  # y_hat: (N, T)

            words.extend(words)
            is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    # gets results and save
    with open("result.txt", 'w') as results:
        # for words, is_heads, tags, y_hat in zip(words, is_heads, Tags, Y_hat):
        for words, is_heads, tags, y_hat in zip(words, Tags, Y_hat):
            #y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [index_to_tag[hat] for hat in y_hat]
            # assert len(preds) == len(words.split()) == len(tags.split())
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                results.write("{} {} {}\n".format(w, t, p))
            results.write("\n")

    # calc metric
    y_true = np.array([tag_to_index[line.split()[1]] for line in open('result.txt', 'r').read().splitlines() if len(line) > 0])
    y_pred = np.array([tag_to_index[line.split()[2]] for line in open('result.txt', 'r').read().splitlines() if len(line) > 0])

    acc = (y_true == y_pred).astype(np.int32).sum() / len(y_true)

    print("acc=%.2f" % acc)


if __name__ == "__main__":
    main()
