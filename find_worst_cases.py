import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import load_train_test_datasets, decode_char
from model import FCN_MLP_Model
from train import LitModel


def load_test_dataset():
    torch.manual_seed(2023)
    return load_train_test_datasets('./captchas', 0.2)[1]


def load_torch_model():
    return LitModel.load_from_checkpoint('model.ckpt', torch_model=FCN_MLP_Model()).torch_model


def main():
    test_dataset = load_test_dataset()
    model = load_torch_model()

    cers = []
    losses = []
    labels = []
    predictions = []

    for x, target in DataLoader(test_dataset):
        y = model(x)
        # TODO: properly reuse metrics calculation
        cers.append(sum((torch.argmax(yi, 1) != target[:,i]).sum() for i, yi in enumerate(y)) / (len(y) * x.size(0)))
        losses.append(sum([F.cross_entropy(yi, target[:,i]) for i, yi in enumerate(y)]) / len(y))
        labels.append(''.join(map(decode_char, target.view(-1))))
        predictions.append(''.join(decode_char(torch.argmax(yi, 1).view(-1)) for yi in y))

    # TODO: parametrize; provide info about mismatches
    worst_cers = torch.topk(torch.tensor(cers), 20)[1]
    worst_losses = torch.topk(torch.tensor(losses), 20)[1]
    print('Worst CER:\n\t', '\n\t'.join(labels[i] + ' (got ' + predictions[i] + ')' for i in worst_cers), sep='')
    print('Worst Loss:\n\t', '\n\t'.join(labels[i] + ' (got ' + predictions[i] +')' for i in worst_losses), sep='')


if __name__ == '__main__':
    main()
