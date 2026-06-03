from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch

from train_classifier3d import TinyClassifier3D

ROOT = Path("generated")
MODEL_PATH = Path("models/thorix_classifier3d_smoke.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()

    df = pd.read_csv(ROOT / "thorix_dataset.csv")
    row = df[df["variant_id"] == args.variant].iloc[0]

    ct = np.load(row["ct_volume_path"]).astype(np.float32)
    ct = np.clip(ct, -1000, 700)
    ct = (ct + 1000) / 1700.0

    x = torch.from_numpy(ct).unsqueeze(0).unsqueeze(0).to(DEVICE)

    model = TinyClassifier3D().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit).item()

    pred = 1 if prob >= 0.5 else 0

    print("Variant:", args.variant)
    print("Actual label:", int(row["classification_label"]))
    print("Predicted label:", pred)
    print("Malignant probability:", prob)
    print("Prediction:", "malignant" if pred == 1 else "benign")


if __name__ == "__main__":
    main()