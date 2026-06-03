import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path("generated")

csv_file = ROOT / "thorix_dataset.csv"

df = pd.read_csv(csv_file)

updated = 0

for idx, row in df.iterrows():

    variant_id = row["variant_id"]

    lesion_file = (
        ROOT
        / variant_id
        / "lesion_metadata.json"
    )

    if not lesion_file.exists():
        continue

    with open(lesion_file) as f:
        lesion = json.load(f)

    center = lesion["center_mm"]

    radius = lesion["radius_mm"]

    volume_mm3 = (
        4.0 / 3.0
        * math.pi
        * radius**3
    )

    df.loc[idx, "lesion_center_x_mm"] = center[0]
    df.loc[idx, "lesion_center_y_mm"] = center[1]
    df.loc[idx, "lesion_center_z_mm"] = center[2]

    df.loc[idx, "lesion_radius_mm"] = radius

    df.loc[idx, "lesion_volume_mm3"] = round(
        volume_mm3,
        2
    )

    updated += 1

df.to_csv(csv_file, index=False)

print(
    f"Updated {updated} lesion records"
)

print(
    f"Saved: {csv_file}"
)