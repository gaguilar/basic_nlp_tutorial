import torch
import torch.nn as nn


class CNNLayer(nn.Module):
    def __init__(self, input_dim, channels, kernels, maxlen):
        super(CNNLayer, self).__init__()

        assert len(kernels) == len(channels)

        self.input_dim = input_dim
        self.maxlen = maxlen  # maximum sequence length
        self.kernels = kernels  # playing the role of n-gram of different orders
        self.channels = channels  # the number of output channels per convolution layer

        self.cnn = {}
        self.bn = {}

        for kernel, out_channels in zip(kernels, channels):
            self.cnn[f'{kernel}_gram'] = nn.Conv1d(self.input_dim, out_channels, kernel)
            self.bn[f'{kernel}_gram'] = nn.BatchNorm1d(out_channels)

        self.cnn = nn.ModuleDict(self.cnn)
        self.bn = nn.ModuleDict(self.bn)

    def forward(self, embeddings):
        batch_size = embeddings.size(0)
        seq_length = embeddings.size(1)
        seq_maxlen = min(seq_length, self.maxlen)

        # Prepare for sliding the Conv1d across time
        embeddings = embeddings.transpose(1, 2)  # -> (batch, embedding, seq_length)

        convs = []
        for kernel, channels in zip(self.kernels, self.channels):
            cnn_key = f'{kernel}_gram'

            convolved = self.cnn[cnn_key](embeddings)  # -> (batch, n_filters, channels)

            curr_shape = convolved.size()
            expt_shape = (batch_size, channels, seq_maxlen - kernel + 1)

            assert curr_shape == expt_shape, "Wrong size: {}. Expected {}".format(curr_shape, expt_shape)

            convolved = self.bn[cnn_key](convolved)  # -> (batch, n_filters, channels)
            convolved, _ = torch.max(convolved, dim=2)  # -> (batch, n_filters)
            convolved = torch.nn.functional.relu(convolved)
            convs.append(convolved)

        convs = torch.cat(convs, dim=1)  # -> (batch, sum(n_filters))  dim 1 is the sum of n_filters from all cnn layers

        return {'output': convs}
