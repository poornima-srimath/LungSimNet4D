import json
import random
from pathlib import Path

import trimesh

random.seed(42)

NUM_VARIANTS = 10

BASE = Path("mesh50XCAT_WholeBody")
OUT = Path("generated")

parts = {
    "Body": "WholeBody_Body.stl",
    "Lung": "WholeBody_Lungs.stl",
    "Skeleton": "WholeBody_Skeleton.stl",
    "Bronchi": "WholeBody_BronchiTree.stl",
    "Liver": "WholeBody_Liver.stl",
    "AirCavity": "WholeBody_AirCavity.stl",
}

OUT.mkdir(exist_ok=True)

for i in range(1, NUM_VARIANTS + 1):

    variant_id = f"variant_{i:04d}"

    variant_dir = OUT / variant_id
    stl_dir = variant_dir / "stl"

    stl_dir.mkdir(parents=True, exist_ok=True)

    body_scale = round(random.uniform(0.94, 1.06), 4)
    lung_scale = round(random.uniform(0.85, 1.15), 4)

    x_scale = round(random.uniform(0.95, 1.05), 4)
    y_scale = round(random.uniform(0.95, 1.05), 4)
    z_scale = round(random.uniform(0.90, 1.10), 4)

    manifest = {
        "variant_id": variant_id,
        "body_scale": body_scale,
        "lung_scale": lung_scale,
        "x_scale": x_scale,
        "y_scale": y_scale,
        "z_scale": z_scale
    }

    for part_name, filename in parts.items():

        mesh = trimesh.load(BASE / filename)

        if part_name in ["Lung", "Bronchi", "AirCavity"]:

            mesh.apply_scale([
                lung_scale * x_scale,
                lung_scale * y_scale,
                lung_scale * z_scale
            ])

        else:

            mesh.apply_scale([
                body_scale * x_scale,
                body_scale * y_scale,
                body_scale * z_scale
            ])

        mesh.export(stl_dir / f"{part_name}.stl")

    with open(variant_dir / "mesh_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Created {variant_id}")

print(f"\nGenerated {NUM_VARIANTS} anatomical variants.")