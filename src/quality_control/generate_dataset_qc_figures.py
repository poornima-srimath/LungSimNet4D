from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT = Path("generated")
RESULTS = Path("results/qc")
RESULTS.mkdir(parents=True, exist_ok=True)

CSV_PATH = ROOT / "thorix_dataset.csv"

df = pd.read_csv(CSV_PATH)

if "status" in df.columns:
    df = df[df["status"] == "READY"].copy()

print("READY samples:", len(df))


def save_bar(series, title, xlabel, ylabel, filename):
    plt.figure(figsize=(7, 5))
    series.value_counts().plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(RESULTS / filename, dpi=300, bbox_inches="tight")
    plt.close()


def save_hist(values, title, xlabel, ylabel, filename, bins=10):
    plt.figure(figsize=(7, 5))
    plt.hist(values.dropna(), bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(RESULTS / filename, dpi=300, bbox_inches="tight")
    plt.close()


# Basic QC
save_hist(df["age"], "Age Distribution", "Age", "Sample Count", "qc_age_distribution.png")
save_bar(df["gender"], "Gender Distribution", "Gender", "Sample Count", "qc_gender_distribution.png")
save_bar(df["lesion_type"], "Lesion Type Distribution", "Lesion Type", "Sample Count", "qc_lesion_type_distribution.png")
save_hist(df["lesion_radius_mm"], "Lesion Radius Distribution", "Radius (mm)", "Sample Count", "qc_lesion_radius_distribution.png")

if "lesion_voxel_count" in df.columns:
    save_hist(
        df["lesion_voxel_count"],
        "Lesion Voxel Count Distribution",
        "Lesion Voxels",
        "Sample Count",
        "qc_lesion_voxel_count_distribution.png",
    )


# Radius vs voxel count
if "lesion_voxel_count" in df.columns:
    plt.figure(figsize=(8, 6))

    for label in sorted(df["classification_label"].unique()):
        sub = df[df["classification_label"] == label]
        name = "Benign" if label == 0 else "Malignant"

        plt.scatter(
            sub["lesion_radius_mm"],
            sub["lesion_voxel_count"],
            label=name,
            alpha=0.8,
        )

    plt.xlabel("Lesion Radius (mm)")
    plt.ylabel("Lesion Voxel Count")
    plt.title("Radius vs Lesion Voxel Count")
    plt.legend()
    plt.grid(True)
    plt.savefig(RESULTS / "qc_radius_vs_voxelcount.png", dpi=300, bbox_inches="tight")
    plt.close()


# 3D lesion spatial distribution
required_spatial = [
    "lesion_center_x_mm",
    "lesion_center_y_mm",
    "lesion_center_z_mm",
]

if all(col in df.columns for col in required_spatial):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    for label in sorted(df["classification_label"].unique()):
        sub = df[df["classification_label"] == label]
        name = "Benign" if label == 0 else "Malignant"

        ax.scatter(
            sub["lesion_center_x_mm"],
            sub["lesion_center_y_mm"],
            sub["lesion_center_z_mm"],
            label=name,
        )

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("THORIX Lesion Spatial Distribution")
    ax.legend()

    plt.savefig(RESULTS / "qc_lesion_spatial_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


# PCA cluster projection
features = [
    "age",
    "lesion_radius_mm",
    "lesion_center_x_mm",
    "lesion_center_y_mm",
    "lesion_center_z_mm",
]

if "lesion_voxel_count" in df.columns:
    features.append("lesion_voxel_count")

if all(col in df.columns for col in features) and len(df) >= 3:
    X = df[features].fillna(0)
    X = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))

    for label in sorted(df["classification_label"].unique()):
        mask = df["classification_label"] == label
        name = "Benign" if label == 0 else "Malignant"

        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            label=name,
            alpha=0.8,
        )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.title("THORIX Dataset PCA Projection")
    plt.legend()
    plt.grid(True)

    plt.savefig(RESULTS / "qc_pca_projection.png", dpi=300, bbox_inches="tight")
    plt.close()


# Correlation heatmap
corr_cols = [
    "age",
    "lesion_radius_mm",
    "lesion_center_x_mm",
    "lesion_center_y_mm",
    "lesion_center_z_mm",
    "classification_label",
]

if "lesion_voxel_count" in df.columns:
    corr_cols.append("lesion_voxel_count")

corr_cols = [col for col in corr_cols if col in df.columns]

if len(corr_cols) >= 3:
    corr = df[corr_cols].corr()

    plt.figure(figsize=(9, 7))
    plt.imshow(corr)
    plt.colorbar()

    plt.xticks(
        range(len(corr.columns)),
        corr.columns,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        range(len(corr.columns)),
        corr.columns,
    )

    plt.title("THORIX Feature Correlation Heatmap")
    plt.tight_layout()

    plt.savefig(RESULTS / "qc_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


# QC summary CSV
summary = {
    "ready_samples": len(df),
    "benign_count": int((df["classification_label"] == 0).sum()),
    "malignant_count": int((df["classification_label"] == 1).sum()),
    "mean_age": float(df["age"].mean()),
    "mean_lesion_radius_mm": float(df["lesion_radius_mm"].mean()),
}

if "lesion_voxel_count" in df.columns:
    summary["mean_lesion_voxel_count"] = float(df["lesion_voxel_count"].mean())

pd.DataFrame([summary]).to_csv(RESULTS / "qc_summary.csv", index=False)

print("Saved QC outputs to:", RESULTS)
print(summary)