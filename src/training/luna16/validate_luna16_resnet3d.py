from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from monai.networks.nets import resnet10

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MANIFEST = Path("luna16_patches/luna16_patch_manifest.csv")
MODEL_PATH = Path("models/luna16_resnet3d.pt")

RESULTS = Path("results/luna16_validation")
RESULTS.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.55


def load_model():
    model = resnet10(
        spatial_dims=3,
        n_input_channels=1,
        num_classes=1,
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    model.eval()
    return model


def predict_patch(model, patch_path):
    patch = np.load(patch_path).astype(np.float32)

    x = (
        torch.from_numpy(patch)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit).item()

    return prob, patch


def save_prediction_image(patch, label, prob, out_path):
    z = patch.shape[0] // 2

    plt.figure(figsize=(6, 6))
    plt.imshow(patch[z], cmap="gray")
    plt.axis("off")

    pred = 1 if prob >= THRESHOLD else 0

    plt.title(
        f"Actual: {label} | Pred: {pred} | Prob: {prob:.3f}"
    )

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    df = pd.read_csv(MANIFEST)

    test_df = df[df["split"] == "test"].copy()

    print("Test samples:", len(test_df))
    print(test_df["label"].value_counts())

    model = load_model()

    y_true = []
    y_prob = []
    y_pred = []

    sample_images_saved = 0

    for idx, row in test_df.iterrows():
        prob, patch = predict_patch(
            model,
            row["patch_path"]
        )

        label = int(row["label"])
        pred = 1 if prob >= THRESHOLD else 0

        y_true.append(label)
        y_prob.append(prob)
        y_pred.append(pred)

        # Save 2 example prediction images
        if sample_images_saved < 2:
            out_file = (
                RESULTS
                / f"luna16_prediction_{sample_images_saved+1}_label_{label}_prob_{prob:.3f}.png"
            )

            save_prediction_image(
                patch,
                label,
                prob,
                out_file
            )

            sample_images_saved += 1

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred)

    print("\n=== LUNA16 VALIDATION ===")
    print("Threshold:", THRESHOLD)
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1       :", f1)
    print("AUC      :", auc)

    pd.DataFrame([{
        "threshold": THRESHOLD,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
    }]).to_csv(
        RESULTS / "luna16_validation_metrics.csv",
        index=False
    )

    # Save confusion matrix
    plt.figure(figsize=(5, 5))
    plt.imshow(cm)
    plt.colorbar()
    plt.title("LUNA16 Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(
        RESULTS / "luna16_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # Save ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("LUNA16 ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        RESULTS / "luna16_roc_curve.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("\nSaved validation outputs to:", RESULTS)


if __name__ == "__main__":
    main()