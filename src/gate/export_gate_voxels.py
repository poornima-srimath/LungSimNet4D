from pathlib import Path
import numpy as np

ROOT = Path("generated")

VALIDATION_VARIANTS = [
    "variant_0001",
    "variant_0004",
    "variant_0005",
]

PITCH_MM = 12


def export_variant(variant):
    ct_path = ROOT / variant / "ct_like" / "ct_like_volume.npy"

    if not ct_path.exists():
        print(f"{variant}: missing ct_like_volume.npy")
        return False

    ct = np.load(ct_path)

    out = ROOT / variant / "gate_voxel"
    out.mkdir(parents=True, exist_ok=True)

    labels = np.zeros(ct.shape, dtype=np.uint16)

    labels[ct == 40] = 1       # Body / Water
    labels[ct == -750] = 2     # Lung
    labels[ct == 150] = 3      # Lesion / Water

    raw_path = out / "phantom.i33"
    hdr_path = out / "phantom.h33"
    range_path = out / "AttnRange.dat"

    labels.tofile(raw_path)

    z, y, x = labels.shape

    header = f"""!INTERFILE :=
!name of data file := generated/{variant}/gate_voxel/phantom.i33
!number of dimensions := 3
!matrix size [1] := {x}
!matrix size [2] := {y}
!matrix size [3] := {z}
!number format := unsigned integer
!number of bytes per pixel := 2
!imagedata byte order := LITTLEENDIAN
!scaling factor (mm/pixel) [1] := {PITCH_MM}
!scaling factor (mm/pixel) [2] := {PITCH_MM}
!scaling factor (mm/pixel) [3] := {PITCH_MM}
!slice thickness (pixels) := {PITCH_MM}
!number of slices := {z}
!END OF INTERFILE :=
"""

    hdr_path.write_text(header)

    # Label to material mapping
    range_text = """0 0 Air
1 1 Water
2 2 Lung
3 3 Water
"""

    range_path.write_text(range_text)

    print(f"{variant}: exported")
    print("  Shape:", labels.shape)
    print("  Labels:", np.unique(labels).tolist())
    print("  Files:", hdr_path, raw_path, range_path)

    return True


def main():
    ok = 0

    for variant in VALIDATION_VARIANTS:
        if export_variant(variant):
            ok += 1

    print(f"\nExported {ok}/{len(VALIDATION_VARIANTS)} variants for GATE validation")


if __name__ == "__main__":
    main()