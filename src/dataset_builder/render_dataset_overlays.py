from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("generated")

variants = sorted(ROOT.glob("variant_*"))

validated = 0

print(f"Found {len(variants)} variants")

for variant_dir in variants:

    variant = variant_dir.name

    try:

        ct_path = (
            variant_dir
            / "ct_like"
            / "ct_like_volume.npy"
        )

        if not ct_path.exists():
            continue

        hu = np.load(ct_path)

        lesion_mask = hu == 150

        lesion_z = np.where(
            lesion_mask.sum(axis=(1, 2)) > 0
        )[0]

        if len(lesion_z) == 0:
            print(
                f"{variant}: no lesion voxels"
            )
            continue

        out_dir = (
            variant_dir
            / "ct_like"
            / "overlay"
        )

        out_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        mid_z = lesion_z[
            len(lesion_z) // 2
        ]

        ct_slice = hu[mid_z]
        mask_slice = lesion_mask[mid_z]

        plt.figure(figsize=(6, 6))

        plt.imshow(
            ct_slice,
            cmap="gray",
            vmin=-1000,
            vmax=700
        )

        overlay = np.zeros(
            (*mask_slice.shape, 4)
        )

        overlay[mask_slice] = [
            1,
            0,
            0,
            0.75
        ]

        plt.imshow(overlay)

        plt.axis("off")

        plt.title(
            f"{variant} z={mid_z}"
        )

        plt.savefig(
            out_dir / "overlay.png",
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        validated += 1

        print(
            f"{variant}: overlay created"
        )

    except Exception as e:

        print(
            f"{variant}: failed {e}"
        )

print(
    f"\nGenerated overlays for "
    f"{validated} variants"
)