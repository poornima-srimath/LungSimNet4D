from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from monai.networks.nets import resnet10

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from torch.utils.data import Dataset, DataLoader


MANIFEST = Path(
    "luna16_patches/luna16_patch_manifest.csv"
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

EPOCHS = 10
BATCH_SIZE = 4
LR = 1e-4

RESULTS = Path("results/luna16")
RESULTS.mkdir(
    parents=True,
    exist_ok=True,
)

MODELS = Path("models")
MODELS.mkdir(
    exist_ok=True
)


class LunaDataset(Dataset):

    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        x = np.load(
            row["patch_path"]
        ).astype(np.float32)

        y = np.float32(
            row["label"]
        )

        return (
            torch.from_numpy(x).unsqueeze(0),
            torch.tensor(y)
        )


def evaluate(model, loader):

    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(DEVICE)

            logits = model(x)

            prob = torch.sigmoid(
                logits
            ).cpu().numpy().flatten()

            pred = (
                prob >= 0.75
            ).astype(int)

            y_true.extend(
                y.numpy().flatten()
            )

            y_pred.extend(pred)

            y_prob.extend(prob)

    acc = accuracy_score(
        y_true,
        y_pred
    )

    prec = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    rec = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_true,
        y_prob
    )

    return (
        acc,
        prec,
        rec,
        f1,
        auc,
        y_true,
        y_pred,
    )


def main():

    df = pd.read_csv(
        MANIFEST
    )

    train_df = df[
        df["split"] == "train"
    ]

    val_df = df[
        df["split"] == "val"
    ]

    test_df = df[
        df["split"] == "test"
    ]

    train_df = train_df.sample(200, random_state=42)
    val_df = val_df.sample(60, random_state=42)
    test_df = test_df.sample(60, random_state=42)

    print(
        "Train:",
        len(train_df)
    )

    print(
        "Val:",
        len(val_df)
    )

    print(
        "Test:",
        len(test_df)
    )

    train_loader = DataLoader(
        LunaDataset(train_df),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        LunaDataset(val_df),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        LunaDataset(test_df),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    epoch_start = time.time()
    model = resnet10(
        spatial_dims=3,
        n_input_channels=1,
        num_classes=1,
    ).to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    history = {
        "epoch": [],
        "loss": [],
        "val_auc": [],
    }

    best_auc = 0

    print(
        "Device:",
        DEVICE
    )

    for epoch in range(EPOCHS):

        model.train()

        epoch_loss = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            if batch_idx % 25 == 0:
                print(
                    f"Epoch {epoch+1}/{EPOCHS} | "
                    f"Batch {batch_idx}/{len(train_loader)}"
                )

            x = x.to(DEVICE)

            y = (
                y.float()
                .unsqueeze(1)
                .to(DEVICE)
            )

            optimizer.zero_grad()

            logits = model(x)

            loss = loss_fn(
                logits,
                y,
            )

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        (
            val_acc,
            val_prec,
            val_rec,
            val_f1,
            val_auc,
            _,
            _,
        ) = evaluate(
            model,
            val_loader
        )

        print(
            f"Epoch {epoch+1:02d} completed in "
            f"{time.time()-epoch_start:.1f}s"
        )

        history["epoch"].append(
            epoch + 1
        )

        history["loss"].append(
            epoch_loss /
            len(train_loader)
        )

        history["val_auc"].append(
            val_auc
        )

        print(
            f"Epoch {epoch+1:02d} | "
            f"Loss {epoch_loss/len(train_loader):.4f} | "
            f"Val AUC {val_auc:.4f}"
        )

        if val_auc > best_auc:

            best_auc = val_auc

            torch.save(
                model.state_dict(),
                MODELS /
                "luna16_resnet3d.pt"
            )

    (
        acc,
        prec,
        rec,
        f1,
        auc,
        y_true,
        y_pred,
    ) = evaluate(
        model,
        test_loader
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("\n=== TEST ===")

    print(
        "Accuracy:",
        acc
    )

    print(
        "Precision:",
        prec
    )

    print(
        "Recall:",
        rec
    )

    print(
        "F1:",
        f1
    )

    print(
        "AUC:",
        auc
    )

    pd.DataFrame(
        [{
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "AUC": auc,
        }]
    ).to_csv(
        RESULTS /
        "metrics.csv",
        index=False
    )

    plt.figure(
        figsize=(7,5)
    )

    plt.plot(
        history["epoch"],
        history["loss"]
    )

    plt.title(
        "Training Loss"
    )

    plt.grid(True)

    plt.savefig(
        RESULTS /
        "loss.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    plt.figure(
        figsize=(7,5)
    )

    plt.plot(
        history["epoch"],
        history["val_auc"]
    )

    plt.title(
        "Validation AUC"
    )

    plt.grid(True)

    plt.savefig(
        RESULTS /
        "auc.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    plt.figure(
        figsize=(5,5)
    )

    plt.imshow(cm)

    plt.colorbar()

    plt.title(
        "Confusion Matrix"
    )

    plt.savefig(
        RESULTS /
        "confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "\nSaved:",
        RESULTS
    )


if __name__ == "__main__":
    main()