from pathlib import Path
import random

import numpy as np
import pandas as pd
import SimpleITK as sitk

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

LUNA_ROOT = Path("luna16")
ANNOTATIONS = LUNA_ROOT / "annotations.csv"

OUT = Path("luna16_patches")

PATCH_SIZE = 64
NEG_PER_POS = 1
MAX_CASES = 366

OUT.mkdir(exist_ok=True)
(OUT / "positive").mkdir(exist_ok=True)
(OUT / "negative").mkdir(exist_ok=True)


def find_scan(seriesuid):
    matches = list(LUNA_ROOT.glob(f"**/{seriesuid}.mhd"))
    return matches[0] if matches else None


def load_scan(path):
    img = sitk.ReadImage(str(path))

    volume = sitk.GetArrayFromImage(img)  # z, y, x
    origin = np.array(img.GetOrigin())    # x, y, z
    spacing = np.array(img.GetSpacing())  # x, y, z

    return volume, origin, spacing


def world_to_voxel(coord_xyz, origin, spacing):
    return np.round(
        (coord_xyz - origin) / spacing
    ).astype(int)

def crop_patch(volume, center_zyx, size=64):

    z, y, x = center_zyx

    half = size // 2

    z = int(np.clip(z, half, volume.shape[0]-half-1))
    y = int(np.clip(y, half, volume.shape[1]-half-1))
    x = int(np.clip(x, half, volume.shape[2]-half-1))

    return volume[
        z-half:z+half,
        y-half:y+half,
        x-half:x+half
    ]

def normalize_hu(patch):
    patch = np.clip(patch, -1000, 400)
    patch = (patch + 1000) / 1400.0
    return patch.astype(np.float32)


def far_from_nodules(candidate, nodule_centers, min_dist=80):
    for center in nodule_centers:
        if np.linalg.norm(candidate - center) < min_dist:
            return False

    return True


def build_scan_level_splits(seriesuids):
    seriesuids = list(seriesuids)

    random.shuffle(seriesuids)

    n = len(seriesuids)

    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    train_ids = seriesuids[:train_end]
    val_ids = seriesuids[train_end:val_end]
    test_ids = seriesuids[val_end:]

    split_rows = []

    for sid in train_ids:
        split_rows.append({
            "seriesuid": sid,
            "split": "train"
        })

    for sid in val_ids:
        split_rows.append({
            "seriesuid": sid,
            "split": "val"
        })

    for sid in test_ids:
        split_rows.append({
            "seriesuid": sid,
            "split": "test"
        })

    split_df = pd.DataFrame(split_rows)

    split_df.to_csv(
        OUT / "luna16_scan_splits.csv",
        index=False
    )

    print("\nScan-level split:")
    print(split_df["split"].value_counts())

    return dict(
        zip(
            split_df["seriesuid"],
            split_df["split"]
        )
    )


def main():
    if not ANNOTATIONS.exists():
        raise FileNotFoundError(
            f"Missing annotations file: {ANNOTATIONS}"
        )

    ann = pd.read_csv(ANNOTATIONS)

    # Only use scans that have annotations
    seriesuids = sorted(
        ann["seriesuid"].unique()
    )

    # Limit to 366 clinical nodule cases
    seriesuids = seriesuids[:MAX_CASES]

    print("Total annotated scans selected:", len(seriesuids))

    split_map = build_scan_level_splits(seriesuids)

    rows = []

    for scan_idx, seriesuid in enumerate(seriesuids):

        scan_path = find_scan(seriesuid)

        if scan_path is None:
            print("Missing scan:", seriesuid)
            continue

        print(
            f"\n[{scan_idx + 1}/{len(seriesuids)}] "
            f"{seriesuid}"
        )

        volume, origin, spacing = load_scan(scan_path)

        group = ann[
            ann["seriesuid"] == seriesuid
        ]

        nodule_voxels = []

        for _, r in group.iterrows():

            coord_xyz = np.array([
                r["coordX"],
                r["coordY"],
                r["coordZ"]
            ])

            voxel_xyz = world_to_voxel(
                coord_xyz,
                origin,
                spacing
            )

            voxel_zyx = np.array([
                voxel_xyz[2],
                voxel_xyz[1],
                voxel_xyz[0]
            ])

            nodule_voxels.append(voxel_zyx)

        split = split_map[seriesuid]

        for nodule_idx, center_zyx in enumerate(nodule_voxels):

            # --------------------------
            # Positive patch
            # --------------------------

            patch = crop_patch(
                volume,
                center_zyx,
                PATCH_SIZE
            )

            patch = normalize_hu(patch)

            pos_name = (
                f"{seriesuid}_pos_{nodule_idx}.npy"
            )

            pos_path = OUT / "positive" / pos_name

            np.save(
                pos_path,
                patch
            )

            rows.append({
                "patch_path": str(pos_path),
                "seriesuid": seriesuid,
                "label": 1,
                "type": "positive",
                "split": split,
                "nodule_index": nodule_idx,
            })

            # --------------------------
            # Negative patches
            # --------------------------

            for neg_idx in range(NEG_PER_POS):

                saved_negative = False

                for _ in range(100):

                    z = random.randint(
                        PATCH_SIZE // 2,
                        volume.shape[0] - PATCH_SIZE // 2 - 1
                    )

                    y = random.randint(
                        PATCH_SIZE // 2,
                        volume.shape[1] - PATCH_SIZE // 2 - 1
                    )

                    x = random.randint(
                        PATCH_SIZE // 2,
                        volume.shape[2] - PATCH_SIZE // 2 - 1
                    )

                    candidate = np.array([z, y, x])

                    if not far_from_nodules(
                        candidate,
                        nodule_voxels
                    ):
                        continue

                    neg_patch = crop_patch(
                        volume,
                        candidate,
                        PATCH_SIZE
                    )

                    neg_patch = normalize_hu(
                        neg_patch
                    )

                    neg_name = (
                        f"{seriesuid}_neg_"
                        f"{nodule_idx}_{neg_idx}.npy"
                    )

                    neg_path = OUT / "negative" / neg_name

                    np.save(
                        neg_path,
                        neg_patch
                    )

                    rows.append({
                        "patch_path": str(neg_path),
                        "seriesuid": seriesuid,
                        "label": 0,
                        "type": "negative",
                        "split": split,
                        "nodule_index": nodule_idx,
                    })

                    saved_negative = True
                    break

                if not saved_negative:
                    print(
                        "Warning: could not create negative patch for",
                        seriesuid,
                        "nodule",
                        nodule_idx
                    )

    manifest = pd.DataFrame(rows)

    manifest.to_csv(
        OUT / "luna16_patch_manifest.csv",
        index=False
    )

    print("\nSaved:", OUT / "luna16_patch_manifest.csv")

    print("\nPatch label counts:")
    print(manifest["label"].value_counts())

    print("\nPatch split counts:")
    print(manifest["split"].value_counts())

    print("\nPatch counts by split and label:")
    print(
        pd.crosstab(
            manifest["split"],
            manifest["label"]
        )
    )


if __name__ == "__main__":
    main()