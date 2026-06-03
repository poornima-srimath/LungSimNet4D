import csv
from pathlib import Path

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEED = 42

ROOT = Path("generated")
CSV_PATH = ROOT / "thorix_dataset_with_splits.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 30
BATCH_SIZE = 1
LR = 1e-3

RESULTS = Path("results/segmentation")
RESULTS.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

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


class SmallUNet3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 8, 3, padding=1),
            nn.ReLU(),
        )

        self.pool = nn.MaxPool3d(2)

        self.enc2 = nn.Sequential(
            nn.Conv3d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, padding=1),
            nn.ReLU(),
        )

        self.up = nn.Upsample(
            scale_factor=2,
            mode="trilinear",
            align_corners=False,
        )

        self.dec1 = nn.Sequential(
            nn.Conv3d(24, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 8, 3, padding=1),
            nn.ReLU(),
        )

        self.out = nn.Conv3d(8, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        up = self.up(e2)

        dz = e1.shape[2] - up.shape[2]
        dy = e1.shape[3] - up.shape[3]
        dx = e1.shape[4] - up.shape[4]

        e1 = e1[
            :,
            :,
            dz // 2 : dz // 2 + up.shape[2],
            dy // 2 : dy // 2 + up.shape[3],
            dx // 2 : dx // 2 + up.shape[4],
        ]

        x = torch.cat([up, e1], dim=1)
        x = self.dec1(x)
        return self.out(x)


def crop_mask(mask, logits):
    _, _, dz, dy, dx = logits.shape
    return mask[:, :, :dz, :dy, :dx]


def metrics(logits, mask, threshold=0.2, eps=1e-6):
    prob = torch.sigmoid(logits)
    pred = (prob > threshold).float()

    tp = (pred * mask).sum()
    fp = (pred * (1 - mask)).sum()
    fn = ((1 - pred) * mask).sum()
    tn = ((1 - pred) * (1 - mask)).sum()

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    return dice.item(), iou.item(), acc.item()


def plot_metric(history, key, title, filename):
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

    model = SmallUNet3D().to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([500.0]).to(DEVICE)
    )

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    history = {
        "epoch": [],
        "loss": [],
        "dice": [],
        "iou": [],
        "accuracy": [],
    }

    print("\nDevice:", DEVICE)

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss = 0
        total_dice = 0
        total_iou = 0
        total_acc = 0

        for ct, mask in loader:
            ct = ct.to(DEVICE)
            mask = mask.to(DEVICE)

            opt.zero_grad()

            logits = model(ct)
            mask = crop_mask(mask, logits)

            loss = loss_fn(logits, mask)
            loss.backward()
            opt.step()

            dice, iou, acc = metrics(logits.detach(), mask)

            total_loss += loss.item()
            total_dice += dice
            total_iou += iou
            total_acc += acc

        n = len(loader)

        row = {
            "epoch": epoch,
            "loss": total_loss / n,
            "dice": total_dice / n,
            "iou": total_iou / n,
            "accuracy": total_acc / n,
        }

        for k in history:
            history[k].append(row[k])

        print(
            f"Epoch {epoch:02d} | "
            f"Loss {row['loss']:.4f} | "
            f"Dice {row['dice']:.4f} | "
            f"IoU {row['iou']:.4f} | "
            f"Acc {row['accuracy']:.4f}"
        )

    pd.DataFrame(history).to_csv(
        RESULTS / "segmentation_metrics.csv",
        index=False,
    )

    plot_metric(history, "loss", "Segmentation Loss", "segmentation_loss.png")
    plot_metric(history, "dice", "Segmentation Dice", "segmentation_dice.png")
    plot_metric(history, "iou", "Segmentation IoU", "segmentation_iou.png")
    plot_metric(history, "accuracy", "Segmentation Accuracy", "segmentation_accuracy.png")

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_dir / "thorix_unet3d_smoke.pt")

    print("\nSaved:")
    print(RESULTS / "segmentation_metrics.csv")
    print(RESULTS / "segmentation_loss.png")
    print(RESULTS / "segmentation_dice.png")
    print(RESULTS / "segmentation_iou.png")
    print(RESULTS / "segmentation_accuracy.png")


if __name__ == "__main__":
    main()