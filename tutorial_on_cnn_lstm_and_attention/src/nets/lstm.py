import torch
import torch.nn as nn


class LSTMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, bidirectional=False, num_layers=1, drop_prob=0.3):
        super(LSTMLayer, self).__init__()

        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim // 2 if bidirectional else hidden_dim,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            dropout=drop_prob if num_layers > 1 else 0,
                            batch_first=True)

    def forward(self, vectors, mask):
        batch_size = vectors.size(0)
        max_length = vectors.size(1)
        lengths = mask.sum(-1)

        lstm_out, _ = self.lstm(vectors)  # (batch, seq_len, num_directions * hidden_size)

        assert lstm_out.size(0) == batch_size
        assert lstm_out.size(1) == max_length
        assert lstm_out.size(2) == self.hidden_dim

        if self.bidirectional:
            # Separate the directions of the LSTM
            lstm_out = lstm_out.view(batch_size, max_length, 2, self.hidden_dim // 2)

            # Pick up the last hidden state per direction
            fw_last_hn = lstm_out[range(batch_size), lengths - 1, 0]  # (batch, hidden // 2)
            bw_last_hn = lstm_out[range(batch_size), 0, 1]            # (batch, hidden // 2)

            last_hn = torch.cat([fw_last_hn, bw_last_hn], dim=1)      # (batch, hidden // 2) -> (batch, hidden)
        else:
            last_hn = lstm_out[range(batch_size), lengths - 1]        # (batch, hidden)

        return {'output': last_hn, 'outputs': lstm_out}

