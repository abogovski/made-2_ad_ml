import os
import glob

import numpy as np
import torch

from PIL import Image, ImageOps

ALPHABET_SIZE = 10 + 26


def encode_char(c):
    assert len(c) == 1 and c.isalnum()
    return ord(c) + (10 - ord('a') if c.isalpha() else - ord('0'))


def decode_char(val):
    assert val >= 0 and val < ALPHABET_SIZE
    return chr(val + (ord('0') if val < 10 else ord('a') - 10))


class CaptchasDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.X, self.y = [], []
        for fname in glob.iglob(os.path.join(path, '*.png')):
            with Image.open(fname) as img:
                x = torch.tensor(np.array(ImageOps.grayscale(img), dtype=np.float32) / 256.)
                self.X.append(x.reshape(1, *x.size()))

            label = os.path.splitext(os.path.basename(fname))[0]
            self.y.append(torch.LongTensor(list(map(encode_char, label))))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_train_test_datasets(dataset_path, test_ratio):
    dataset = CaptchasDataset(dataset_path)
    test_length = int(np.ceil(len(dataset) * test_ratio))  # NOTE: can't use fractions
    return torch.utils.data.random_split(dataset, [len(dataset) - test_length, test_length])


if __name__ == '__main__':
    train_dataset, test_dataset = load_train_test_datasets('./captchas', 0.2)
    print(len(train_dataset), len(test_dataset))
