import os
import torch
import torch.nn as nn
import numpy as np


class WordEmbedder(nn.Module):
    def __init__(self, vocab, glove_file):
        super(WordEmbedder, self).__init__()
        assert os.path.exists(glove_file) and glove_file.endswith('.txt'), glove_file

        self.emb_dim = None

        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'

        index_to_word = [self.PAD_TOKEN, self.UNK_TOKEN]
        index_to_vect = [None, None]

        with open(glove_file, 'r') as fp:
            for line in fp:
                line = line.split()

                if line[0] not in vocab:
                    continue

                w = line[0]
                v = np.array([float(value) for value in line[1:]])

                if self.emb_dim is None:
                    self.emb_dim = v.shape[0]

                index_to_word.append(w)
                index_to_vect.append(v)

        index_to_vect[0] = np.zeros(self.emb_dim)
        index_to_vect[1] = np.mean(index_to_vect[2:], axis=0)

        self.embeddings = torch.from_numpy(np.array(index_to_vect)).float()
        self.embeddings = nn.Embedding.from_pretrained(self.embeddings, freeze=False)

        self.index_to_word = {i: w for i, w in enumerate(index_to_word)}
        self.word_to_index = {w: i for i, w in self.index_to_word.items()}

    def forward(self, samples):
        pad_ix = self.word_to_index[self.PAD_TOKEN]
        unk_ix = self.word_to_index[self.UNK_TOKEN]

        maxlen = max([len(s) for s in samples])

        encoded = [[self.word_to_index.get(token, unk_ix) for token in tokens] for tokens in samples]
        masks = torch.zeros(len(samples), maxlen).long()

        # Padding and masking
        for i in range(len(encoded)):
            masks[i, :len(encoded[i])] = 1
            encoded[i] += [pad_ix] * max(0, (maxlen - len(encoded[i])))

        encoded = torch.tensor(encoded).long()

        if torch.cuda.is_available():
            encoded = encoded.cuda()
            masks = masks.cuda()

        result = {
            'output': self.embeddings(encoded),
            'mask': masks,
            'encoded': encoded
        }

        return result
