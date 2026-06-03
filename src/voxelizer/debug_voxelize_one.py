from pathlib import Path
import numpy as np
import trimesh
import matplotlib.pyplot as plt

variant = "variant_0001"
PITCH = 12.0   # try 6 first; if lesion still 0, try 4.0

base = Path(f"generated/{variant}/stl")
out = Path(f"generated/{variant}/ct_like_debug")
out.mkdir(parents=True, exist_ok=True)

body_mesh = trimesh.load(base / "Body.stl")
lung_mesh = trimesh.load(base / "Lung.stl")
lesion_mesh = trimesh.load(base / "Lesion.stl")

print("\n==== MESH DEBUG ====")
print("Variant:", variant)
print("Pitch:", PITCH)

print("\nBody bounds:")
print(body_mesh.bounds)

print("\nLung bounds:")
print(lung_mesh.bounds)

print("\nLesion bounds:")
print(lesion_mesh.bounds)

print("\nLesion centroid:")
print(lesion_mesh.centroid)

radius_est = np.max(
    np.linalg.norm(
        lesion_mesh.vertices - lesion_mesh.centroid,
        axis=1
    )
)

print("\nLesion radius estimate:")
print(radius_est)

# common world bounds from body
min_b, max_b = body_mesh.bounds

xs = np.arange(min_b[0], max_b[0], PITCH)
ys = np.arange(min_b[1], max_b[1], PITCH)
zs = np.arange(min_b[2], max_b[2], PITCH)

shape = (len(zs), len(ys), len(xs))

print("\n==== GRID DEBUG ====")
print("Grid shape:", shape)
print("X count:", len(xs), "range:", xs[0], xs[-1])
print("Y count:", len(ys), "range:", ys[0], ys[-1])
print("Z count:", len(zs), "range:", zs[0], zs[-1])
print("Total voxels:", np.prod(shape))

hu = np.full(shape, -1000, dtype=np.int16)

def mask_from_mesh(mesh, label):
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
            print(label, "z", zi, "/", len(zs))

    print(f"{label} voxel count:", int(mask.sum()))
    return mask

print("\n==== VOXELIZING ====")

body = mask_from_mesh(body_mesh, "body")
lung = mask_from_mesh(lung_mesh, "lung")
lesion = mask_from_mesh(lesion_mesh, "lesion")

print("\n==== MASK COUNTS ====")
print("Body voxels:", int(body.sum()))
print("Lung voxels:", int(lung.sum()))
print("Lesion voxels:", int(lesion.sum()))

hu[body] = 40
hu[lung] = -750
hu[lesion] = 150

np.save(out / "ct_like_volume.npy", hu)
np.save(out / "lesion_mask.npy", lesion.astype(np.uint8))
np.save(out / "lung_mask.npy", lung.astype(np.uint8))

lesion_z = np.where(lesion.sum(axis=(1, 2)) > 0)[0]

print("\n==== LESION SLICE DEBUG ====")

if len(lesion_z) == 0:
    print("NO LESION VOXELS FOUND.")
    print("Try reducing PITCH to 4.0 or increasing minimum lesion radius.")
else:
    print("Lesion appears in z slices:", lesion_z.min(), "to", lesion_z.max())

    z_index = int(lesion_z[len(lesion_z) // 2])

    plt.figure(figsize=(6, 6))
    plt.imshow(hu[z_index], cmap="gray", vmin=-1000, vmax=700)

    overlay = np.zeros((*lesion[z_index].shape, 4))
    overlay[lesion[z_index]] = [1, 0, 0, 0.75]
    plt.imshow(overlay)

    plt.axis("off")
    plt.title(f"{variant} lesion overlay z={z_index}")
    plt.savefig(out / "overlay_debug.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved:", out / "overlay_debug.png")

print("\nSaved debug volume to:", out)