import argparse
import datetime
from pathlib import Path
from typing import List, Tuple

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Timer
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor, nn
from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import FakeData, ImageFolder
from torchvision.transforms import v2

from src.modelzoo import create_model

NUM_CLASSES = 1000


class ImageNetDataModule(L.LightningDataModule):
    def __init__(
        self, data_dir: str = "./datasets", batch_size: int = 128, num_workers: int = 8
    ) -> None:
        super().__init__()
        self.num_classes = NUM_CLASSES
        self.img_shape = (3, 224, 224)

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_tfms = v2.Compose(
            [
                v2.PILToTensor(),
                v2.Resize(256, antialias=True),
                v2.RandAugment(),
                v2.RandomCrop(224),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.test_tfms = v2.Compose(
            [
                v2.PILToTensor(),
                v2.Resize(256, antialias=True),
                v2.CenterCrop(224),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def setup(self, stage: str) -> None:
        self.train_set = FakeData(num_classes=NUM_CLASSES, transform=self.train_tfms)
        self.test_set = FakeData(num_classes=NUM_CLASSES, transform=self.test_tfms)
        # self.train_set = ImageFolder(
        # str(self.data_dir / "train"),
        # transform=self.train_tfms,
        # )
        # self.test_set = ImageFolder(
        # str(self.data_dir / "val"),
        # transform=self.test_tfms,
        # )

    def train_dataloader(self) -> None:
        cutmix = v2.CutMix(num_classes=NUM_CLASSES)
        mixup = v2.MixUp(num_classes=NUM_CLASSES)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

        def collate_fn(batch):
            return cutmix_or_mixup(*default_collate(batch))

        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            persistent_workers=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            persistent_workers=True,
            num_workers=self.num_workers,
        )


class ClassifierModule(L.LightningModule):
    def __init__(self, model_name: str, max_epochs: int) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = create_model(model_name)
        self.criterion = nn.CrossEntropyLoss()
        self.max_epochs = max_epochs

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(
        self, train_batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        x, y = train_batch
        out = self.forward(x)

        loss = self.criterion(out, y)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, val_batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        x, y = val_batch
        out = self.forward(x)
        loss = self.criterion(out, y)
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_acc", (torch.argmax(out, 1) == y).float().mean(), sync_dist=True)

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LRScheduler]]:
        optimizer = torch.optim.SGD(
            self.parameters(), lr=1e-1, nesterov=True, momentum=0.9, weight_decay=1e-4
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1,
        )
        return [optimizer], [lr_scheduler]

    def load_state_dict(self, state_dict) -> None:
        d = {k.removeprefix("model."): v for k, v in state_dict.items()}
        self.model.load_state_dict(d, strict=False)


def main(hparams):
    # create run name
    L.seed_everything(hparams.seed, workers=True)

    run_name = f"{hparams.model}_{hparams.seed}"

    # look for existing checkpoint to resume from
    ckpt_path = Path("./checkpoints", run_name, "last.ckpt")
    if ckpt_path.exists():
        ckpt_path = str(ckpt_path)
    else:
        ckpt_path = None

    trainer = L.Trainer(
        accelerator="gpu",
        devices=hparams.devices,
        precision=hparams.precision,
        strategy="ddp",
        gradient_clip_val=hparams.gradient_clip_val,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        logger=WandbLogger(
            project="ImageNet1k",
            name=run_name,
            id=run_name,
        ),
        callbacks=[
            ModelCheckpoint(
                dirpath=f"./checkpoints/{run_name}",
                filename="{epoch:02d}-{val_loss:.2f}",
                monitor="val_loss",
                mode="min",
                save_last=True,
            ),
        ],
    )

    model = ClassifierModule(hparams.model, hparams.max_epochs)
    datamodule = ImageNetDataModule(
        hparams.data_dir, hparams.batch_size, hparams.num_workers
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./")
    parser.add_argument("--model", type=str, default="c1resnet18")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=90)

    args = parser.parse_args()

    main(args)
