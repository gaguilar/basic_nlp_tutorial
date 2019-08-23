import os
import copy
import random

from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset


class IMDBDatset(Dataset):
    def __init__(self, datadir, maxlen=64):
        assert os.path.exists(datadir), datadir

        self.tokens = []
        self.labels = []

        self.maxlen = maxlen
        self.label_to_index = {'pos': 1, 'neg': 0}

        pos_tokens, pos_labels = read_files(datadir, 'pos', maxlen)
        neg_tokens, neg_labels = read_files(datadir, 'neg', maxlen)

        self.tokens.extend(pos_tokens + neg_tokens)
        self.labels.extend(pos_labels + neg_labels)

        self.tokens, self.labels = shuffle(self.tokens, self.labels)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, item):
        return self.tokens[item], self.label_to_index[self.labels[item]]

    def split_data(self, size):
        dataset = copy.deepcopy(self)
        dataset.tokens = dataset.tokens[-size:]
        dataset.labels = dataset.labels[-size:]

        self.tokens = self.tokens[:-size]
        self.labels = self.labels[:-size]

        return dataset


def read_files(datadir, sentiment, maxlen):
    sent_dir = os.path.join(datadir, sentiment)

    tokens = [word_tokenize(open(os.path.join(sent_dir, sent_file)).read())[:maxlen]
              for sent_file in os.listdir(sent_dir)
              if sent_file.endswith('.txt')]

    labels = [sentiment] * len(tokens)

    return tokens, labels


def shuffle(tokens, labels):
    z = list(zip(tokens, labels))
    random.shuffle(z)
    return zip(*z)
