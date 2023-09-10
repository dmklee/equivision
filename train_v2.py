import argparse
import math
from pathlib import Path
from typing import List, Tuple

import lightning as L
import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor, nn
from torch.utils.data import DataLoader, default_collate
from torchvision.datasets import FakeData, ImageFolder
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

from src.modelzoo import create_model


class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU).
    Heavily based on 'torch.utils.data.DistributedSampler'.

    This is taken from Pytorch repo:
        https://github.com/pytorch/vision/blob/main/references/classification/sampler.py
        which borrowed from the DeiT Repo:
    https://github.com/facebookresearch/deit/blob/main/samplers.py
    """

    def __init__(
        self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, repetitions=3
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available!")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available!")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.dataset) * float(repetitions) / self.num_replicas)
        )
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = int(
            math.floor(len(self.dataset) // 256 * 256 / self.num_replicas)
        )
        self.shuffle = shuffle
        self.seed = seed
        self.repetitions = repetitions

    def __iter__(self):
        if self.shuffle:
            # Deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(self.repetitions)]
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[: self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class ImageNetDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./datasets",
        batch_size: int = 128,
        num_workers: int = 8,
        aug_repeats: int = 0,
        dummy: bool = False,
    ) -> None:
        super().__init__()
        self.img_shape = (3, 224, 224)
        self.dummy = dummy

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.aug_repeats = aug_repeats

        self.train_tfms = v2.Compose(
            [
                v2.RandomResizedCrop(176, antialias=True),
                v2.RandomHorizontalFlip(0.5),
                v2.TrivialAugmentWide(interpolation=InterpolationMode.BILINEAR),
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                v2.RandomErasing(p=0.1),
                v2.ToPureTensor(),
            ]
        )

        self.test_tfms = v2.Compose(
            [
                v2.Resize(232, antialias=True),
                v2.CenterCrop(224),
                v2.PILToTensor(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                v2.ToPureTensor(),
            ]
        )

    def setup(self, stage: str) -> None:
        if self.dummy:
            print("[WARNING] Using dummy data!!")
            self.train_set = FakeData(
                1281167, (3, 224, 224), num_classes=1000, transform=self.train_tfms
            )
            self.test_set = FakeData(
                50000, (3, 224, 224), num_classes=1000, transform=self.test_tfms
            )
        else:
            self.train_set = ImageFolder(
                str(self.data_dir / "train"),
                transform=self.train_tfms,
            )
            self.test_set = ImageFolder(
                str(self.data_dir / "val"),
                transform=self.test_tfms,
            )

    def train_dataloader(self) -> None:
        cutmix = v2.CutMix(num_classes=1000)
        mixup = v2.MixUp(num_classes=1000, alpha=0.2)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

        def collate_fn(batch):
            return cutmix_or_mixup(*default_collate(batch))

        if self.aug_repeats == 0:
            sampler = None
            shuffle = True
        else:
            sampler = RASampler(self.train_set, num_replicas=4)
            shuffle = None

        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            persistent_workers=True,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            persistent_workers=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class ClassifierModule(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        max_epochs: int,
        warmup_epochs: int,
        lr: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        group, layers = model_name.split("resnet")
        self.model = create_model(group, int(layers))
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.lr = lr

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
        norm_params = []
        non_norm_params = []
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                if "bn" in name:
                    norm_params.append(p)
                else:
                    non_norm_params.append(p)

        assert len(norm_params + non_norm_params) == len(list(self.model.parameters()))

        param_groups = [
            {"params": non_norm_params, "weight_decay": 0.00002},
            {"params": norm_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.SGD(param_groups, lr=self.lr, momentum=0.9)
        warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=self.warmup_epochs
        )
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs - self.warmup_epochs, eta_min=0.0
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[self.warmup_epochs],
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

    if hparams.no_wandb:
        logger = None
    else:
        logger = WandbLogger(
            project="ImageNet1k_v2",
            name=run_name,
            id=run_name,
        )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=hparams.devices,
        precision=hparams.precision,
        strategy="ddp",
        max_epochs=hparams.max_epochs,
        # sync_batchnorm=True,
        accumulate_grad_batches=hparams.effective_batch_size // hparams.batch_size,
        logger=logger,
        callbacks=[
            ModelCheckpoint(
                dirpath=f"./checkpoints/{run_name}",
                filename="{epoch:02d}-{val_loss:.2f}",
                monitor="val_loss",
                mode="min",
                save_last=True,
            ),
            LearningRateMonitor(),
            StochasticWeightAveraging(
                swa_lrs=0.05,
                swa_epoch_start=int(0.75 * hparams.max_epochs),
            ),
        ],
    )

    model = ClassifierModule(
        hparams.model, hparams.max_epochs, hparams.warmup_epochs, hparams.lr
    )

    datamodule = ImageNetDataModule(
        data_dir=hparams.data_dir,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        dummy=hparams.dummy_data,
        aug_repeats=hparams.aug_repeats,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    # https://github.com/pytorch/vision/issues/3995#issuecomment-1013906621
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./")
    parser.add_argument("--model", type=str, default="c1resnet18")
    parser.add_argument("--devices", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--effective_batch_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--aug_repeats", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=600)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--lr", type=int, default=0.5)
    parser.add_argument("--dummy_data", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")

    args = parser.parse_args()

    main(args)
    # TODO:
    # - repeated augmentations; check it is uniquely sampling indices each epoch
    # - fix sync batch norm for equivariant models
