"""train.py
===============
Script d’entraînement U‑Net (signal spatial)

*Lancement direct :*
    python train.py

Les chemins CSV sont déjà définis.
—
Paramètres par défaut pensés pour **générer ≈ 1 150 patches**
par split :
    grid_size  = 300 × 300
    patch_size = 32   (multiple de 32 → stride total ResNet OK)
    stride     = 8    (75 % de recouvrement)
Avec la formule ⌊(300−32)/8⌋+1 = 34 positions par axe → 34² = 1 156.

(Le patch 32×32 sort un feature‑map 1×1, donc on garde batch_size ≥2 pour
que BatchNorm reste valide.)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import transforms
import torchvision.transforms.functional as F
import random
import segmentation_models_pytorch as smp

from metrics import RegressionMetrics
from trainer import RegressionTrainer
from dataset import SpatialSignalDataset
from wandb_logger import WandBLogger

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
DEFAULT_TRAIN_CSV = (
    r"C:\\Users\\natha\\PycharmProjects\\ComputerOrientedProject"
    r"\\DataAcquisition\\processed_data\\train.csv"
)
DEFAULT_VAL_CSV = (
    r"C:\\Users\\natha\\PycharmProjects\\ComputerOrientedProject"
    r"\\DataAcquisition\\processed_data\\validation.csv"
)
DEFAULT_TEST_CSV = (
    r"C:\\Users\\natha\\PycharmProjects\\ComputerOrientedProject"
    r"\\DataAcquisition\\processed_data\\test.csv"
)

# -----------------------------------------------------------------------------
# 2 Data‑loader helper
# -----------------------------------------------------------------------------

class SimpleTensorAug:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() < 0.5:
            x = F.hflip(x)
        if random.random() < 0.5:
            x = F.vflip(x)
        if random.random() < 0.5:
            k = random.choice([1, 2, 3])  # 90°, 180°, 270°
            x = torch.rot90(x, k, dims=[1, 2])
        return x

def load_datasets(train_csv: str, val_csv: str, test_csv: str,
                  grid_size: tuple[int, int], patch_size: int, stride: int):
    transform_train = SimpleTensorAug()
    transform_val   = transforms.Compose([])

    common = dict(grid_size=grid_size, patch_size=patch_size, stride=stride)
    return (
        SpatialSignalDataset(train_csv, transform=transform_train, **common),
        SpatialSignalDataset(val_csv,   transform=transform_val,   **common),
        SpatialSignalDataset(test_csv,  transform=transform_val,   **common),
    )

# -----------------------------------------------------------------------------
# 3  Train loop
# -----------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    train_ds, val_ds, _ = load_datasets(
        args.train_csv,
        args.val_csv,
        args.test_csv,
        grid_size=(args.grid_h, args.grid_w),
        patch_size=args.patch_size,
        stride=args.stride,
    )

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=args.lr_gamma)
    loss_fn   = nn.MSELoss()

    train_metric, val_metric = RegressionMetrics(), RegressionMetrics()
    logger = WandBLogger(model=model, run_name=args.run_name)

    trainer = RegressionTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        lr_scheduler=scheduler,
        train_metric=train_metric,
        val_metric=val_metric,
        train_data=train_ds,
        val_data=val_ds,
        device=device,
        num_epochs=args.epochs,
        training_save_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        val_frequency=5,
        logger=logger,
    )
    trainer.train(); logger.finish()

# -----------------------------------------------------------------------------
# 4  CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Train U‑Net for spatial signal regression")

    # CSV paths
    p.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV)
    p.add_argument("--val-csv",   default=DEFAULT_VAL_CSV)
    p.add_argument("--test-csv",  default=DEFAULT_TEST_CSV)

    # Grid / patch params (defaults bumped)
    p.add_argument("--grid-size", nargs=2, type=int, default=[300, 300])
    p.add_argument("--patch-size", type=int, default=32)
    p.add_argument("--stride",     type=int, default=8)

    # Optim / runtime
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs",     type=int, default=50)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--lr-gamma",   type=float, default=0.95)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--gpu-id",     type=str,  default="0")
    p.add_argument("--output-dir", type=str,  default="saved_models")
    p.add_argument("--run-name",   type=str,  default="Unet-run")

    a = p.parse_args(); a.grid_h, a.grid_w = map(int, a.grid_size); return a

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    cli = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = cli.gpu_id
    train(cli)
