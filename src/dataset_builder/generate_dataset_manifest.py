import csv
import json
import random
from pathlib import Path

random.seed(42)

NUM_VARIANTS = 10

OUT = Path("generated")

csv_rows = []

for i in range(1, NUM_VARIANTS + 1):

    variant_id = f"variant_{i:04d}"
    sample_id = f"THX-{i:06d}"

    variant_dir = OUT / variant_id

    gender = random.choice([
        "Male",
        "Female"
    ])

    age = random.choices(
        population=[20, 30, 40, 50, 60, 70],
        weights=[10, 15, 25, 25, 15, 10],
        k=1
    )[0]

    lesion_type = random.choice([
        "benign",
        "malignant"
    ])

    lesion_radius_mm = round(
        random.uniform(3, 15),
        2
    )

    lesion_hu = (
        random.randint(80, 120)
        if lesion_type == "benign"
        else random.randint(150, 280)
    )

    lesion_x = round(
        random.uniform(-60, 60),
        2
    )

    lesion_y = round(
        random.uniform(-40, 40),
        2
    )

    lesion_z = round(
        random.uniform(-80, 80),
        2
    )

    classification_label = (
        0 if lesion_type == "benign" else 1
    )

    variant_config = {

        "sample_id": sample_id,
        "variant_id": variant_id,

        "gender": gender,
        "age": age,

        "classification_label": classification_label,

        "lesion": {

            "enabled": True,

            "type": lesion_type,

            "radius_mm": lesion_radius_mm,

            "x_mm": lesion_x,
            "y_mm": lesion_y,
            "z_mm": lesion_z,

            "hu": lesion_hu
        },

        "dataset": {

            "ct_volume": "ct_like_volume.npy",

            "lesion_mask": "lesion_mask.npy",

            "gate": {

                "phantom_header": "phantom.h33",
                "phantom_data": "phantom.i33",
                "attn_range": "AttnRange.dat"
            }
        }
    }

    # print(f"Generated config for {variant_id}:")
    with open(
        variant_dir / "variant_config.json",
        "w"
    ) as f:

        json.dump(
            variant_config,
            f,
            indent=2
        )

    csv_rows.append({

        "sample_id": sample_id,
        "variant_id": variant_id,

        "gender": gender,
        "age": age,

        "lesion_type": lesion_type,

        "lesion_radius_mm": lesion_radius_mm,

        "lesion_hu": lesion_hu,

        "lesion_x_mm": lesion_x,
        "lesion_y_mm": lesion_y,
        "lesion_z_mm": lesion_z,

        "classification_label":
            classification_label
    })

csv_path = OUT / "thorix_dataset.csv"

with open(
    csv_path,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=csv_rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(csv_rows)

print(
    f"Generated dataset manifest: {csv_path}"
)