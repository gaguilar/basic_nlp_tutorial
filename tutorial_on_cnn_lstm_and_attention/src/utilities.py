import torch
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler


def process_logits(logits):
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    preds = np.round(probs).tolist()
    probs = probs.tolist()
    return preds, probs


def count_params(model):
    return sum([p.nelement() for p in model.parameters() if p.requires_grad])


def flatten(posts):
    return [t for tokens in posts for t in tokens]


def collate(batch):
    tokens, labels = zip(*batch)
    targets = torch.tensor(labels, dtype=torch.long)
    return tokens, targets


def get_dataloader(dataset, batch_size, shuffle=False):
    sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
    dloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate)
    return dloader


def track_best_model(model_path, model, epoch, best_acc, dev_acc, dev_loss):
    if best_acc > dev_acc:
        return best_acc, ''
    state = {
        'epoch': epoch,
        'acc': dev_acc,
        'loss': dev_loss,
        'model': model.state_dict()
    }
    torch.save(state, model_path)
    return dev_acc, ' * '


def train(model, dataloaders, optimizer, config):
    best_acc = 0
    for epoch in range(1, config['epochs'] + 1):
        epoch_msg = f'E{epoch:03d}'
        epoch_track = ''

        for dataset in dataloaders:
            if dataset == 'train':
                model.train()
                model.zero_grad()
            else:
                model.eval()

            epoch_loss = 0
            preds, truth = [], []

            # ========================================================================
            for batch_i, (tokens, targets) in enumerate(dataloaders[dataset]):
                result = model(tokens, targets)
                loss = result['loss']

                if dataset == 'train':
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    model.zero_grad()

                epoch_loss += loss.item() * len(targets)
                batch_preds, _ = process_logits(result['output'])

                preds += batch_preds
                truth += targets.data.cpu().tolist()
            # ========================================================================

            epoch_acc = accuracy_score(truth, preds)
            epoch_loss /= len(dataloaders[dataset].dataset)
            epoch_msg += ' [{}] Loss: {:.4f}, Acc: {:.4f}'.format(dataset.upper(), epoch_loss, epoch_acc)

            if dataset == 'dev':
                best_acc, epoch_track = track_best_model(config['checkpoint'], model, epoch, best_acc, epoch_acc,
                                                         epoch_loss)

        print(epoch_msg + epoch_track)
    print("Done training!")

    state = torch.load(config['checkpoint'])
    model.load_state_dict(state['model'])

    print('Returning best model from epoch {} with loss {:.5f} and accuracy {:.5f}'.format(
        state['epoch'], state['loss'], state['acc']))
    return model

