import json
import random
from pathlib import Path

import numpy as np
import trimesh

random.seed(42)
np.random.seed(42)

ROOT = Path("generated")

variants = sorted(ROOT.glob("variant_*"))

print(f"Found {len(variants)} variants")

for variant_dir in variants:

    variant = variant_dir.name

    try:

        lung_path = variant_dir / "stl" / "Lung.stl"

        lung = trimesh.load(lung_path)

        bounds = lung.bounds
        min_b, max_b = bounds

        points = np.random.uniform(
            min_b,
            max_b,
            size=(5000, 3)
        )

        inside = lung.contains(points)
        inside_points = points[inside]

        if len(inside_points) == 0:
            print(f"Skipping {variant}: no interior points")
            continue

        center = inside_points[
            random.randint(
                0,
                len(inside_points) - 1
            )
        ]

        lesion_type = random.choice([
            "benign",
            "malignant"
        ])

        radius_mm = (
            round(random.uniform(14, 20), 2)
            if lesion_type == "benign"
            else round(random.uniform(20, 30), 2)
        )

        lesion = trimesh.creation.icosphere(
            subdivisions=3,
            radius=radius_mm
        )

        lesion.apply_translation(center)

        lesion.export(
            variant_dir / "stl" / "Lesion.stl"
        )

        meta = {

            "variant_id": variant,

            "lesion_type": lesion_type,

            "center_mm": center.tolist(),

            "radius_mm": radius_mm,

            "method": "sampled_inside_lung_mesh"
        }

        with open(
            variant_dir / "lesion_metadata.json",
            "w"
        ) as f:

            json.dump(
                meta,
                f,
                indent=2
            )

        print(
            f"{variant} -> "
            f"{lesion_type} "
            f"{radius_mm} mm"
        )

    except Exception as e:

        print(
            f"{variant} failed: {e}"
        )