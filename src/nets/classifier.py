import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, embedder, extractor):
        super(LSTMClassifier, self).__init__()
        self.embedder = embedder
        self.extractor = extractor
        self.classifier = nn.Linear(extractor.hidden_dim, 1)
        self.xentropy = nn.BCEWithLogitsLoss()

    def forward(self, tokens, targets=None):
        embedded = self.embedder(tokens)
        extracted = self.extractor(embedded['output'], embedded['mask'])

        logits = self.classifier(extracted['output'])
        loss = None

        if targets is not None:
            logits = logits.view(-1)
            targets = targets.float()
            loss = self.xentropy(logits, targets)

        return {'output': logits, 'loss': loss}


class LSTMAttentionClassifier(nn.Module):
    def __init__(self, embedder, extractor, attention):
        super(LSTMAttentionClassifier, self).__init__()
        self.embedder = embedder
        self.extractor = extractor
        self.attention = attention
        self.classifier = nn.Linear(extractor.hidden_dim, 1)
        self.xentropy = nn.BCEWithLogitsLoss()

    def forward(self, tokens, targets=None):
        embedded = self.embedder(tokens)
        extracted = self.extractor(embedded['output'], embedded['mask'])
        attended = self.attention(extracted['outputs'], embedded['mask'])

        logits = self.classifier(attended['output'])
        loss = None

        if targets is not None:
            logits = logits.view(-1)
            targets = targets.float()
            loss = self.xentropy(logits, targets)

        return {'output': logits, 'loss': loss, 'attentions': attended['attentions']}
