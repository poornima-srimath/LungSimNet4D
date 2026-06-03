from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from train_unet3d import SmallUNet3D, crop_mask, metrics

ROOT = Path("generated")
MODEL_PATH = Path("models/thorix_unet3d_smoke.pt")
OUT = Path("results/validation")
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_sample(variant_id):
    row = pd.read_csv(ROOT / "thorix_dataset.csv")
    row = row[row["variant_id"] == variant_id].iloc[0]

    ct = np.load(row["ct_volume_path"]).astype(np.float32)
    mask = np.load(row["lesion_mask_path"]).astype(np.float32)

    ct_norm = np.clip(ct, -1000, 700)
    ct_norm = (ct_norm + 1000) / 1700.0

    return ct, ct_norm, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, help="Example: variant_0001")
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()

    ct_raw, ct_norm, mask = load_sample(args.variant)

    model = SmallUNet3D().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    x = torch.from_numpy(ct_norm).unsqueeze(0).unsqueeze(0).to(DEVICE)
    y = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        y = crop_mask(y, logits)
        prob = torch.sigmoid(logits)

    dice, iou, acc = metrics(logits, y, threshold=args.threshold)

    pred = (prob > args.threshold).float().cpu().numpy()[0, 0]
    gt = y.cpu().numpy()[0, 0]
    ct_crop = ct_raw[: pred.shape[0], : pred.shape[1], : pred.shape[2]]

    lesion_z = np.where(gt.sum(axis=(1, 2)) > 0)[0]
    z = int(lesion_z[len(lesion_z) // 2]) if len(lesion_z) else pred.shape[0] // 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(ct_crop[z], cmap="gray", vmin=-1000, vmax=700)
    axes[0].set_title("CT Slice")
    axes[0].axis("off")

    axes[1].imshow(ct_crop[z], cmap="gray", vmin=-1000, vmax=700)
    axes[1].imshow(gt[z], alpha=0.7)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(ct_crop[z], cmap="gray", vmin=-1000, vmax=700)
    axes[2].imshow(pred[z], alpha=0.7)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    out_file = OUT / f"{args.variant}_segmentation_prediction.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

    print("Variant:", args.variant)
    print("Dice:", dice)
    print("IoU:", iou)
    print("Accuracy:", acc)
    print("Saved:", out_file)


if __name__ == "__main__":
    main()