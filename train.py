import argparse
from pathlib import Path
from typing import List, Tuple

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor, nn
from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import ImageFolder
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
                v2.RandomResizedCrop(224, antialias=True),
                v2.RandomHorizontalFlip(0.5),
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                v2.ToPureTensor(),
            ]
        )

        self.test_tfms = v2.Compose(
            [
                v2.Resize(256, antialias=True),
                v2.CenterCrop(224),
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                v2.ToPureTensor(),
            ]
        )

    def setup(self, stage: str) -> None:
        # self.train_set = FakeData(num_classes=NUM_CLASSES, transform=self.train_tfms)
        # self.test_set = FakeData(num_classes=NUM_CLASSES, transform=self.test_tfms)
        self.train_set = ImageFolder(
            str(self.data_dir / "train"),
            transform=self.train_tfms,
        )
        self.test_set = ImageFolder(
            str(self.data_dir / "val"),
            transform=self.test_tfms,
        )

    def train_dataloader(self) -> None:
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
    def __init__(
        self,
        model_name: str,
        max_epochs: int = 90,
        lr: float = 0.1,
        lr_step_size: int = 30,
        lr_gamma: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = create_model(model_name)
        self.criterion = nn.CrossEntropyLoss()
        self.max_epochs = max_epochs
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.momentum = momentum
        self.weight_decay = weight_decay

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

        pred = out.topk(k=5, dim=1)[1].t()
        correct = pred.eq(y.view(1, -1).expand_as(pred))
        acc1 = correct[0].float().mean()
        acc5 = correct[:5].any(dim=0).float().mean()
        loss = self.criterion(out, y)
        self.log("val_loss", loss, sync_dist=True)
        self.log("val_acc@1", acc1, sync_dist=True)
        self.log("val_acc@5", acc5, sync_dist=True)

    def configure_optimizers(
        self,
    ) -> Tuple[
        List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]
    ]:
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.lr_step_size,
            gamma=self.lr_gamma,
        )
        return [optimizer], [lr_scheduler]

    def load_state_dict(self, state_dict) -> None:
        d = {k.removeprefix("model."): v for k, v in state_dict.items()}
        self.model.load_state_dict(d, strict=False)


def main(hparams):
    # create run name
    L.seed_everything(hparams.seed, workers=True)

    project = "ImageNet1k_v1"
    run_name = f"{hparams.model}_{hparams.seed}"

    # look for existing checkpoint to resume from
    ckpt_path = Path("./checkpoints", project, run_name, "last.ckpt")
    if ckpt_path.exists():
        ckpt_path = str(ckpt_path)
    else:
        ckpt_path = None

    trainer = L.Trainer(
        accelerator="gpu",
        devices=hparams.devices,
        precision=hparams.precision,
        strategy="ddp",
        max_epochs=hparams.max_epochs,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        logger=WandbLogger(
            project=project,
            name=run_name,
            id=run_name,
        ),
        callbacks=[
            ModelCheckpoint(
                dirpath=f"./checkpoints/{project}/{run_name}",
                filename="{epoch:02d}-{val_loss:.2f}",
                monitor="val_loss",
                mode="min",
                save_last=True,
            ),
        ],
    )

    model = ClassifierModule(
        model_name=hparams.model,
        max_epochs=hparams.max_epochs,
        lr=hparams.lr,
        lr_step_size=hparams.lr_step_size,
        lr_gamma=hparams.lr_gamma,
        momentum=hparams.momentum,
        weight_decay=hparams.weight_decay,
    )
    datamodule = ImageNetDataModule(
        hparams.data_dir, hparams.batch_size, hparams.num_workers
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    # https://github.com/pytorch/vision/tree/main/references/classification#resnet

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./")
    parser.add_argument("--model", type=str, default="c1resnet18")
    parser.add_argument("--devices", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr_step_size", type=int, default=30)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--max_epochs", type=int, default=90)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    args = parser.parse_args()

    main(args)
