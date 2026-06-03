from pathlib import Path
import numpy as np
import trimesh
import matplotlib.pyplot as plt

ROOT = Path("generated")
PITCH = 12.0

variants = sorted(ROOT.glob("variant_*"))

print(f"Found {len(variants)} variants")

def mask_from_mesh(mesh, xs, ys, zs, shape, label):
    mask = np.zeros(shape, dtype=bool)

    xx, yy = np.meshgrid(xs, ys, indexing="xy")

    for zi, z in enumerate(zs):
        pts = np.column_stack([
            xx.ravel(),
            yy.ravel(),
            np.full(xx.size, z)
        ])

        inside = mesh.contains(pts).reshape(len(ys), len(xs))
        mask[zi] = inside

        if zi % 30 == 0:
            print(f"  {label} z {zi}/{len(zs)}")

    return mask

done = 0
failed = 0

for variant_dir in variants:
    variant = variant_dir.name
    base = variant_dir / "stl"
    out = variant_dir / "ct_like"
    out.mkdir(parents=True, exist_ok=True)

    try:
        body_path = base / "Body.stl"
        lung_path = base / "Lung.stl"
        lesion_path = base / "Lesion.stl"

        if not body_path.exists() or not lung_path.exists() or not lesion_path.exists():
            print(f"{variant}: missing STL, skipping")
            failed += 1
            continue

        print(f"\nVoxelizing {variant}")

        body_mesh = trimesh.load(body_path)
        lung_mesh = trimesh.load(lung_path)
        lesion_mesh = trimesh.load(lesion_path)

        min_b, max_b = body_mesh.bounds

        xs = np.arange(min_b[0], max_b[0], PITCH)
        ys = np.arange(min_b[1], max_b[1], PITCH)
        zs = np.arange(min_b[2], max_b[2], PITCH)

        shape = (len(zs), len(ys), len(xs))
        print("  Grid shape:", shape)

        hu = np.full(shape, -1000, dtype=np.int16)

        body = mask_from_mesh(body_mesh, xs, ys, zs, shape, "body")
        lung = mask_from_mesh(lung_mesh, xs, ys, zs, shape, "lung")
        lesion = mask_from_mesh(lesion_mesh, xs, ys, zs, shape, "lesion")

        hu[body] = 40
        hu[lung] = -750
        hu[lesion] = 150

        np.save(out / "ct_like_volume.npy", hu)
        np.save(out / "lesion_mask.npy", lesion.astype(np.uint8))
        np.save(out / "lung_mask.npy", lung.astype(np.uint8))

        lesion_z = np.where(lesion.sum(axis=(1, 2)) > 0)[0]

        if len(lesion_z) > 0:
            z_index = int(lesion_z[len(lesion_z) // 2])

            plt.figure(figsize=(5, 5))
            plt.imshow(hu[z_index], cmap="gray", vmin=-1000, vmax=700)

            overlay = np.zeros((*lesion[z_index].shape, 4))
            overlay[lesion[z_index]] = [1, 0, 0, 0.75]
            plt.imshow(overlay)

            plt.axis("off")
            plt.title(f"{variant} lesion overlay z={z_index}")
            plt.savefig(out / "overlay.png", dpi=150, bbox_inches="tight")
            plt.close()

        else:
            print(f"  WARNING: no lesion voxels found")

        done += 1
        print(f"{variant}: done")

    except Exception as e:
        failed += 1
        print(f"{variant}: failed -> {e}")

print("\nBatch voxelization complete")
print("Done:", done)
print("Failed:", failed)