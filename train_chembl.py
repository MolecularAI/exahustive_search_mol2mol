import os
import pickle
import sys

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.strategies.ddp import DDPStrategy
import numpy as np
import lightning as pl
import torch
import yaml
from lib.dataset.pair_dataset import FastPairedDataset
from lib.model.model import LitMolformer


if __name__ == "__main__":
    hparams = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)
    with open(hparams["vocabulary"], "rb") as ifile:
        vocabulary = pickle.load(ifile)

    train = np.load(hparams["train_smiles_path"], allow_pickle=True)[()]
    valid = np.load(hparams["valid_smiles_path"], allow_pickle=True)[()]
    test = np.load(hparams["test_smiles_path"], allow_pickle=True)[()]

    checkpoint_path = sys.argv[2]
    os.makedirs(os.path.join(checkpoint_path, "chkpts"), exist_ok=True)

    print(f"All the checkpoints will be saved in: {checkpoint_path}")

    with open(os.path.join(checkpoint_path, "config.yml"), "w") as yf:
        yaml.dump(hparams, yf)
    model = LitMolformer(**hparams)
    vocabulary = model.vocabulary
    tokenizer = model.tokenizer

    train_dataset = FastPairedDataset(
        train,
        max_length=hparams["max_sequence_length"],
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        drop_last=True,
        collate_fn=FastPairedDataset.collate_fn,
        num_workers=4,
        persistent_workers=True,
    )

    valid_dataset = FastPairedDataset(
        valid,
        max_length=hparams["max_sequence_length"],
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=128,
        shuffle=False,
        drop_last=False,
        collate_fn=FastPairedDataset.collate_fn,
        num_workers=1,
        persistent_workers=True,
    )

    early_stopping_cp = EarlyStopping(
        monitor="valid_loss", min_delta=0.00, patience=200, verbose=True, mode="min"
    )

    callback_cp = ModelCheckpoint(
        dirpath=os.path.join(checkpoint_path, "chkpts"),
        save_top_k=100,
        monitor="valid_loss",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=int(sys.argv[3]),
        max_epochs=5000,
        log_every_n_steps=1,
        callbacks=[early_stopping_cp, callback_cp],
        strategy=DDPStrategy(find_unused_parameters=False),
    )
    trainer.fit(model, train_dataloader, valid_dataloader)
    # To restart the training add:
    # , ckpt_path="results/ecfp4/chkpts/epoch=4-step=499650.ckpt")
    trainer.save_checkpoint(os.path.join(checkpoint_path, "chkpts", "last.ckpt"))
