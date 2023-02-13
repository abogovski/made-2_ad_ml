import argparse
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from dataset import load_train_test_datasets
from model import FCN_MLP_Model


class LitModel(pl.LightningModule):
    def __init__(self, torch_model):
        super().__init__()
        self.torch_model = torch_model

    def training_step(self, batch, batch_idx):
        x, target = batch
        return sum([F.cross_entropy(z, target[:,i]) for i, z in enumerate(self.torch_model(x))]) / target.size(1)

    def validation_step(self, batch, batch_idx):
        x, target = batch
        y = self.torch_model(x)
        val_loss = sum([F.cross_entropy(yi, target[:,i]) for i, yi in enumerate(y)]) / len(y)
        val_cer = sum((torch.argmax(yi, 1) != target[:,i]).sum() for i, yi in enumerate(y)) / (len(y) * x.size(0))

        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_cer', val_cer, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-from-checkpoint', type=str)
    args = parser.parse_args()

    torch.manual_seed(2023)
    train_dataset, test_dataset = load_train_test_datasets('./captchas', 0.2)
    model = LitModel(FCN_MLP_Model())

    trainer = pl.Trainer(
        callbacks=[
            pl.callbacks.EarlyStopping(monitor='val_cer', min_delta=0.00, patience=25, verbose=True, mode='min'),
            pl.callbacks.ModelCheckpoint(save_top_k=3, monitor='val_cer'),
        ],
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    trainer.fit(model, DataLoader(train_dataset, batch_size=32), DataLoader(train_dataset, batch_size=2048))


if __name__ == '__main__':
    main()
