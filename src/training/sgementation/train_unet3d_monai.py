from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from monai.networks.nets import UNet
from torch.utils.data import Dataset, DataLoader

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path("generated")
CSV_PATH = ROOT / "thorix_dataset_with_splits.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 30
BATCH_SIZE = 1
LR = 1e-3
THRESHOLD = 0.2

RESULTS = Path("results/segmentation")
RESULTS.mkdir(parents=True, exist_ok=True)


class ThorixSegDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        ct = np.load(row["ct_volume_path"]).astype(np.float32)
        mask = np.load(row["lesion_mask_path"]).astype(np.float32)

        ct = np.clip(ct, -1000, 700)
        ct = (ct + 1000) / 1700.0

        return (
            torch.from_numpy(ct).unsqueeze(0),
            torch.from_numpy(mask).unsqueeze(0),
        )


def crop_to_match(mask, logits):
    _, _, dz, dy, dx = logits.shape
    return mask[:, :, :dz, :dy, :dx]


def calc_metrics(logits, mask, threshold=THRESHOLD, eps=1e-6):
    pred = (torch.sigmoid(logits) > threshold).float()

    tp = (pred * mask).sum()
    fp = (pred * (1 - mask)).sum()
    fn = ((1 - pred) * mask).sum()
    tn = ((1 - pred) * (1 - mask)).sum()

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    return dice.item(), iou.item(), acc.item()


def plot(history, key, title, filename):
    plt.figure(figsize=(7, 5))
    plt.plot(history["epoch"], history[key], marker="o")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(key)
    plt.grid(True)
    plt.savefig(RESULTS / filename, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    df = pd.read_csv(CSV_PATH)
    df = df[df["status"] == "READY"].copy()

    dataset = ThorixSegDataset(df)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    total_lesion = 0
    total_voxels = 0

    for _, mask in loader:
        total_lesion += mask.sum().item()
        total_voxels += mask.numel()

    print("==== DATASET STATS ====")
    print("Samples:", len(dataset))
    print("Lesion voxels:", int(total_lesion))
    print("Total voxels:", int(total_voxels))
    print("Positive ratio:", total_lesion / total_voxels)

    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64),
        strides=(2, 2),
        num_res_units=2,
    ).to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([500.0]).to(DEVICE)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = {
        "epoch": [],
        "loss": [],
        "dice": [],
        "iou": [],
        "accuracy": [],
    }

    print("Device:", DEVICE)

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss = 0
        total_dice = 0
        total_iou = 0
        total_acc = 0

        for ct, mask in loader:
            ct = ct.to(DEVICE)
            mask = mask.to(DEVICE)

            optimizer.zero_grad()

            logits = model(ct)
            mask = crop_to_match(mask, logits)

            loss = loss_fn(logits, mask)
            loss.backward()
            optimizer.step()

            dice, iou, acc = calc_metrics(logits.detach(), mask)

            total_loss += loss.item()
            total_dice += dice
            total_iou += iou
            total_acc += acc

        n = len(loader)

        history["epoch"].append(epoch)
        history["loss"].append(total_loss / n)
        history["dice"].append(total_dice / n)
        history["iou"].append(total_iou / n)
        history["accuracy"].append(total_acc / n)

        print(
            f"Epoch {epoch:02d} | "
            f"Loss {history['loss'][-1]:.4f} | "
            f"Dice {history['dice'][-1]:.4f} | "
            f"IoU {history['iou'][-1]:.4f} | "
            f"Acc {history['accuracy'][-1]:.4f}"
        )

    pd.DataFrame(history).to_csv(
        RESULTS / "segmentation_metrics.csv",
        index=False,
    )

    plot(history, "loss", "MONAI 3D U-Net Segmentation Loss", "segmentation_loss.png")
    plot(history, "dice", "MONAI 3D U-Net Dice", "segmentation_dice.png")
    plot(history, "iou", "MONAI 3D U-Net IoU", "segmentation_iou.png")
    plot(history, "accuracy", "MONAI 3D U-Net Accuracy", "segmentation_accuracy.png")

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    torch.save(
        model.state_dict(),
        model_dir / "thorix_monai_unet3d.pt",
    )

    print("Saved model:", model_dir / "thorix_monai_unet3d.pt")


if __name__ == "__main__":
    main()