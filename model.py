import pytorch_lightning as pl
from torch import nn
from dataset import ALPHABET_SIZE

NUM_CHARS = 5


class FCN_MLP_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fcn = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 4, (3, 3)), nn.MaxPool2d((2, 2)),
            nn.Conv2d(4, 16, (3, 3)), nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (3, 3)), nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, (3, 3)), nn.MaxPool2d((2, 2)),
            nn.Flatten(),
        )
        self.mlps = [
            nn.Sequential(
                nn.Linear(640, 320),
                nn.ReLU(),
                nn.Linear(320, 160),
                nn.ReLU(),
                nn.Linear(160, ALPHABET_SIZE),
            ) for _ in range(NUM_CHARS)
        ]

    def forward(self, X):
        features = self.fcn(X)
        return [mlp(features) for mlp in self.mlps]
