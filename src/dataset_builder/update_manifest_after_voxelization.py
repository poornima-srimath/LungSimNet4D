from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("generated")
CSV_PATH = ROOT / "thorix_dataset.csv"

df = pd.read_csv(CSV_PATH)

updated = 0
ready = 0
failed = 0

for idx, row in df.iterrows():
    variant_id = row["variant_id"]
    variant_dir = ROOT / variant_id

    ct_path = variant_dir / "ct_like" / "ct_like_volume.npy"
    lesion_mask_path = variant_dir / "ct_like" / "lesion_mask.npy"
    lung_mask_path = variant_dir / "ct_like" / "lung_mask.npy"
    overlay_path = variant_dir / "ct_like" / "overlay.png"

    df.loc[idx, "ct_volume_path"] = str(ct_path)
    df.loc[idx, "lesion_mask_path"] = str(lesion_mask_path)
    df.loc[idx, "lung_mask_path"] = str(lung_mask_path)
    df.loc[idx, "overlay_path"] = str(overlay_path)

    if (
        ct_path.exists()
        and lesion_mask_path.exists()
        and lung_mask_path.exists()
    ):
        try:
            ct = np.load(ct_path)
            lesion_mask = np.load(lesion_mask_path)
            lung_mask = np.load(lung_mask_path)

            df.loc[idx, "voxel_shape_z"] = ct.shape[0]
            df.loc[idx, "voxel_shape_y"] = ct.shape[1]
            df.loc[idx, "voxel_shape_x"] = ct.shape[2]

            df.loc[idx, "lesion_voxel_count"] = int(lesion_mask.sum())
            df.loc[idx, "lung_voxel_count"] = int(lung_mask.sum())

            df.loc[idx, "ct_min_hu"] = int(ct.min())
            df.loc[idx, "ct_max_hu"] = int(ct.max())

            if int(lesion_mask.sum()) > 0:
                df.loc[idx, "status"] = "READY"
                ready += 1
            else:
                df.loc[idx, "status"] = "NO_LESION_VOXELS"
                failed += 1

            updated += 1

        except Exception as e:
            df.loc[idx, "status"] = f"FAILED_LOAD: {e}"
            failed += 1

    else:
        missing = []

        if not ct_path.exists():
            missing.append("ct_like_volume.npy")

        if not lesion_mask_path.exists():
            missing.append("lesion_mask.npy")

        if not lung_mask_path.exists():
            missing.append("lung_mask.npy")

        df.loc[idx, "status"] = "MISSING_" + "_".join(missing)
        failed += 1

df.to_csv(CSV_PATH, index=False)

print("Voxelization manifest update complete")
print("Updated:", updated)
print("READY:", ready)
print("Failed / incomplete:", failed)
print("Saved:", CSV_PATH)