from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from torch.utils.data import Dataset, DataLoader

SEED = 42

ROOT = Path("generated")
CSV_PATH = ROOT / "thorix_dataset_with_splits.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 30
BATCH_SIZE = 1
LR = 1e-3

RESULTS = Path("results/classification")
RESULTS.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

class ThorixClassificationDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        ct = np.load(row["ct_volume_path"]).astype(np.float32)

        ct = np.clip(ct, -1000, 700)
        ct = (ct + 1000) / 1700.0

        label = np.float32(row["classification_label"])

        return (
            torch.from_numpy(ct).unsqueeze(0),
            torch.tensor(label),
        )


class TinyClassifier3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool3d(1),
        )

        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def main():
    df = pd.read_csv(CSV_PATH)
    df = df[df["status"] == "READY"].copy()

    dataset = ThorixClassificationDataset(df)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = TinyClassifier3D().to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    history = {
        "epoch": [],
        "loss": [],
        "accuracy": [],
    }

    print("Device:", DEVICE)
    print("Samples:", len(dataset))

    for epoch in range(EPOCHS):

        model.train()

        epoch_loss = 0

        y_true = []
        y_pred = []

        for ct, label in loader:

            ct = ct.to(DEVICE)

            label = (
                label
                .float()
                .unsqueeze(1)
                .to(DEVICE)
            )

            optimizer.zero_grad()

            logits = model(ct)

            loss = loss_fn(logits, label)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

            preds = (
                torch.sigmoid(logits) > 0.5
            ).float()

            y_true.extend(
                label.cpu().numpy().flatten()
            )

            y_pred.extend(
                preds.cpu().numpy().flatten()
            )

        acc = accuracy_score(y_true, y_pred)

        history["epoch"].append(epoch + 1)
        history["loss"].append(epoch_loss / len(loader))
        history["accuracy"].append(acc)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Loss {epoch_loss / len(loader):.4f} | "
            f"Acc {acc:.4f}"
        )

    # Final evaluation on current dataset
    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():

        for ct, label in loader:

            ct = ct.to(DEVICE)

            logits = model(ct)

            pred = (
                torch.sigmoid(logits) > 0.5
            ).float()

            all_true.extend(
                label.numpy().flatten()
            )

            all_pred.extend(
                pred.cpu().numpy().flatten()
            )

    acc = accuracy_score(all_true, all_pred)

    prec = precision_score(
        all_true,
        all_pred,
        zero_division=0,
    )

    rec = recall_score(
        all_true,
        all_pred,
        zero_division=0,
    )

    f1 = f1_score(
        all_true,
        all_pred,
        zero_division=0,
    )

    cm = confusion_matrix(
        all_true,
        all_pred,
    )

    print("\n=== FINAL METRICS ===")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1       :", f1)

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    torch.save(
        model.state_dict(),
        model_dir / "thorix_classifier3d_smoke.pt",
    )

    print(
        "Saved model:",
        model_dir / "thorix_classifier3d_smoke.pt",
    )

    pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
            ],
            "Value": [
                acc,
                prec,
                rec,
                f1,
            ],
        }
    ).to_csv(
        RESULTS / "classification_metrics.csv",
        index=False,
    )

    plt.figure(figsize=(7, 5))
    plt.plot(
        history["epoch"],
        history["loss"],
    )
    plt.title("Classification Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(
        RESULTS / "classification_loss.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(
        history["epoch"],
        history["accuracy"],
    )
    plt.title("Classification Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(
        RESULTS / "classification_accuracy.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.imshow(cm)
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.savefig(
        RESULTS / "classification_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("\nSaved:")
    print(RESULTS)


if __name__ == "__main__":
    main()