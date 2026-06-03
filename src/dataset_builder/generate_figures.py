from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

variant = "variant_0001"

ct_path = Path(
    f"generated/{variant}/ct_like/ct_like_volume.npy"
)

out = Path(
    f"generated/{variant}/final_revision_figures"
)

out.mkdir(parents=True, exist_ok=True)

hu = np.load(ct_path)

lesion = hu == 150

lesion_z = np.where(
    lesion.sum(axis=(1,2)) > 0
)[0]

if len(lesion_z) == 0:
    raise RuntimeError("No lesion found")

mid_z = int(
    lesion_z[len(lesion_z)//2]
)

# --------------------------
# Axial series
# --------------------------

for offset in [-2,-1,0,1,2]:

    z = mid_z + offset

    if z < 0 or z >= hu.shape[0]:
        continue

    plt.figure(figsize=(6,6))

    plt.imshow(
        hu[z],
        cmap="gray",
        vmin=-1000,
        vmax=700
    )

    plt.axis("off")

    plt.title(
        f"Axial z={z}"
    )

    plt.savefig(
        out / f"axial_z_{z}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# --------------------------
# Orthogonal views
# --------------------------

views = {

    "axial_mid":
        hu[mid_z],

    "coronal_mid":
        hu[:, hu.shape[1]//2, :],

    "sagittal_mid":
        hu[:, :, hu.shape[2]//2],
}

for name, img in views.items():

    plt.figure(figsize=(6,6))

    plt.imshow(
        img,
        cmap="gray",
        vmin=-1000,
        vmax=700
    )

    plt.axis("off")

    plt.title(name)

    plt.savefig(
        out / f"{name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

# --------------------------
# Lesion overlay
# --------------------------

plt.figure(figsize=(6,6))

plt.imshow(
    hu[mid_z],
    cmap="gray",
    vmin=-1000,
    vmax=700
)

overlay = np.zeros(
    (*lesion[mid_z].shape,4)
)

overlay[lesion[mid_z]] = [
    1,0,0,0.8
]

plt.imshow(overlay)

plt.axis("off")

plt.title(
    f"Lesion Overlay z={mid_z}"
)

plt.savefig(
    out / "lesion_overlay.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------
# Maximum intensity projection
# --------------------------

mip = hu.max(axis=0)

plt.figure(figsize=(6,6))

plt.imshow(
    mip,
    cmap="gray"
)

plt.axis("off")

plt.title("MIP Projection")

plt.savefig(
    out / "mip_projection.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved figures:", out)