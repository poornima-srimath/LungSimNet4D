from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path("generated")
CSV_PATH = ROOT / "thorix_dataset.csv"

df = pd.read_csv(CSV_PATH)

df = df[df["status"] == "READY"].copy()

print("READY samples:", len(df))

if len(df) < 5:
    raise RuntimeError("Not enough READY samples to split.")

# 70 / 15 / 15 split
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["classification_label"]
    if df["classification_label"].nunique() > 1 else None
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["classification_label"]
    if temp_df["classification_label"].nunique() > 1 and len(temp_df) >= 4 else None
)

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

split_df = pd.concat([train_df, val_df, test_df]).sort_values("sample_id")

split_df.to_csv(ROOT / "thorix_dataset_with_splits.csv", index=False)

train_df.to_csv(ROOT / "train.csv", index=False)
val_df.to_csv(ROOT / "val.csv", index=False)
test_df.to_csv(ROOT / "test.csv", index=False)

print("Saved:")
print(ROOT / "thorix_dataset_with_splits.csv")
print(ROOT / "train.csv")
print(ROOT / "val.csv")
print(ROOT / "test.csv")

print("\nSplit counts:")
print(split_df["split"].value_counts())

print("\nLabel counts by split:")
print(pd.crosstab(split_df["split"], split_df["classification_label"]))