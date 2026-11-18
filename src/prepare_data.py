"""Prepare train/val/test splits from a CFPB complaints export.

Put the CCDB export (.zip or .csv, exactly as downloaded) in data/ and run:
    python src/prepare_data.py
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 42
TEXT_COL = "Consumer complaint narrative"
LABEL_COL = "Product"
SPLIT_FILES = {"train.csv", "val.csv", "test.csv"}


def find_raw_file():
    """Return the raw export in data/ (any .zip or .csv that is not a split file)."""
    candidates = sorted(
        p for p in DATA.iterdir()
        if p.suffix.lower() in {".csv", ".zip"} and p.name not in SPLIT_FILES
    )
    if not candidates:
        raise FileNotFoundError(
            f"No .zip or .csv found in {DATA}. Put the CFPB export there first."
        )
    if len(candidates) > 1:
        print("Several files found, using the first one:", [c.name for c in candidates])
    return candidates[0]


def main(raw_path, sample_size, min_class_count):
    raw_path = Path(raw_path) if raw_path else find_raw_file()
    print("Reading:", raw_path.name)

    df = pd.read_csv(raw_path, usecols=[LABEL_COL, TEXT_COL])
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df = df.rename(columns={TEXT_COL: "text", LABEL_COL: "label"})
    df["label"] = df["label"].str.strip()

    # Drop very rare classes
    counts = df["label"].value_counts()
    keep = counts[counts >= min_class_count].index
    df = df[df["label"].isin(keep)]

    # Sample to keep training fast
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=SEED)

    train, temp = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=SEED
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["label"], random_state=SEED
    )

    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(DATA / f"{name}.csv", index=False)
        print(f"{name}: {len(part):,} rows")

    print("\nClass distribution (train):")
    print(train["label"].value_counts())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default=None, help="optional path; default: auto-detect in data/")
    p.add_argument("--sample_size", type=int, default=100_000)
    p.add_argument("--min_class_count", type=int, default=500)
    args = p.parse_args()
    main(args.raw, args.sample_size, args.min_class_count)
