"""
34-Class Preprocessing Pipeline for CICIoT2023
Completely separate from the 2-class and 8-class pipelines.
DO NOT modify preprocess-new.py or train-new.py.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# ====================== CONFIGURATION ======================
BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data" / "raw"
OUTPUT_DIR  = BASE_DIR / "data" / "processed_data_34"
MODELS_DIR  = BASE_DIR / "models" / "preprocessing"

RANDOM_STATE = 42
TARGET_TOTAL = 1_000_000
MIN_SAMPLES  = 2_000
MAX_CAP      = 40_000


# ==============================================================
# STEP 1 -- LOAD DATA
# Read all Merged*.csv files. Label is already a column inside
# each file -- no folder iteration or two-pass column discovery
# needed.
# ==============================================================
def load_all_data(data_dir: Path) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 1: LOADING DATA")
    print("=" * 60)

    csv_files = sorted(data_dir.glob("Merged*.csv"))
    print(f"Found {len(csv_files)} merged CSV files in {data_dir}")

    all_dfs = []
    total_rows = 0
    for csv_file in csv_files:
        try:
            chunk = pd.read_csv(csv_file, low_memory=False)
            # Drop any NaN labels immediately (malformed rows)
            chunk = chunk.dropna(subset=["Label"])
            all_dfs.append(chunk)
            total_rows += len(chunk)
            print(f"  {csv_file.name}: {len(chunk):,} rows")
        except Exception as exc:
            print(f"  WARNING: failed to load {csv_file.name}: {exc}")

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nCombined shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Unique labels found: {sorted(df['Label'].unique())}")
    return df


# ==============================================================
# STEP 2 -- DATA CLEANING
# Replace ?inf -> NaN, impute NaN with per-column median,
# drop constant columns (only one unique value).
# ==============================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 2: DATA CLEANING")
    print("=" * 60)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Replace inf / -inf
    inf_count = np.isinf(df[numeric_cols].values).sum()
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    print(f"Replaced {inf_count:,} inf/-inf values with NaN")

    # Fill NaN with column median
    nan_count = int(df[numeric_cols].isnull().sum().sum())
    for col in numeric_cols:
        if df[col].isnull().any():
            med = df[col].median()
            df[col] = df[col].fillna(0.0 if pd.isna(med) else med)
    print(f"Imputed {nan_count:,} NaN values using per-column median")

    # Drop constant columns
    constant = [c for c in numeric_cols if df[c].nunique() <= 1]
    if constant:
        df = df.drop(columns=constant)
        print(f"Dropped {len(constant)} constant column(s): {constant}")
    else:
        print("No constant columns found")

    print(f"Shape after cleaning: {df.shape[0]:,} x {df.shape[1]}")
    return df


# ==============================================================
# STEP 3 -- DATA TYPE OPTIMIZATION
# Downcast all numeric columns to float32 to halve memory usage.
# ==============================================================
def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 3: DTYPE OPTIMIZATION")
    print("=" * 60)

    mb_before = df.memory_usage(deep=True).sum() / 1024 ** 2
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].astype(np.float32)
    mb_after = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"Memory: {mb_before:.0f} MB  ->  {mb_after:.0f} MB")
    return df


# ==============================================================
# STEP 4 -- CLASS DISTRIBUTION ANALYSIS
# Print per-class counts and percentages.
# ==============================================================
def print_class_distribution(df: pd.DataFrame, title: str) -> pd.Series:
    print(f"\n{title}")
    print("-" * 55)
    counts = df["Label"].value_counts().sort_index()
    total  = len(df)
    for cls, cnt in counts.items():
        print(f"  {cls:<38} {cnt:>10,}  ({cnt / total * 100:5.2f}%)")
    print(f"  {'TOTAL':<38} {total:>10,}")
    return counts


# ==============================================================
# STEP 5 & 6 -- SMART CLASS BALANCING + SAMPLING
# Proportionally scale each class toward TARGET_TOTAL,
# clip between MIN_SAMPLES and MAX_CAP, never oversample.
# ==============================================================
def smart_balance(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 5-6: SMART CLASS BALANCING & SAMPLING")
    print("=" * 60)

    class_counts = df["Label"].value_counts()
    total_rows   = len(df)
    scale        = TARGET_TOTAL / total_rows

    print(f"Original rows : {total_rows:,}")
    print(f"Target rows   : {TARGET_TOTAL:,}")
    print(f"Scale factor  : {scale:.4f}")
    print(f"Min samples   : {MIN_SAMPLES:,}  |  Max cap: {MAX_CAP:,}")

    # Proportional cap, clipped, then bounded by actual count (no oversampling)
    raw_caps = (class_counts * scale).round().astype(int)
    caps     = raw_caps.clip(lower=MIN_SAMPLES, upper=MAX_CAP)
    caps     = caps.combine(class_counts, min)   # never exceed real count

    print(f"\n  {'Class':<38} {'Available':>12} {'Cap':>8} {'Sampled':>8}")
    print(f"  {'-'*38} {'-'*12} {'-'*8} {'-'*8}")
    for cls in sorted(class_counts.index):
        avail = class_counts[cls]
        cap   = caps[cls]
        print(f"  {cls:<38} {avail:>12,} {cap:>8,} {min(avail, cap):>8,}")

    sampled = []
    for label, group in df.groupby("Label"):
        cap = int(caps[label])
        if len(group) <= cap:
            sampled.append(group)
        else:
            sampled.append(group.sample(n=cap, random_state=RANDOM_STATE))

    df_balanced = pd.concat(sampled, ignore_index=True)
    print(f"\nBalanced dataset: {len(df_balanced):,} rows")
    return df_balanced


# ==============================================================
# STEP 7 -- SHUFFLE
# Randomise row order so batches drawn during training are i.i.d.
# ==============================================================
def shuffle_data(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 7: SHUFFLING")
    print("=" * 60)

    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Shuffled {len(df):,} rows  (random_state={RANDOM_STATE})")
    return df


# ==============================================================
# STEP 8 -- SPLIT DATA
# Separate X / y, encode labels, stratified 80/10/10 split.
# Encoder saved so inference can map predicted integers back
# to attack-type strings without re-loading the training data.
# ==============================================================
def split_data(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("STEP 8: TRAIN / VAL / TEST SPLIT")
    print("=" * 60)

    feature_cols = [c for c in df.columns if c != "Label"]
    X    = df[feature_cols]
    y_raw = df["Label"]

    le = LabelEncoder()
    y  = le.fit_transform(y_raw)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(le, MODELS_DIR / "label_encoder_34.pkl")
    print(f"Label encoder saved  ->  {MODELS_DIR / 'label_encoder_34.pkl'}")
    print(f"Classes ({len(le.classes_)}): {list(le.classes_)}")

    # 80 % train  /  10 % val  /  10 % test  (stratified)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    total = len(X)
    print(f"\nSplit sizes:")
    print(f"  Train : {len(X_train):>10,}  ({len(X_train)/total*100:.1f} %)")
    print(f"  Val   : {len(X_val):>10,}  ({len(X_val)/total*100:.1f} %)")
    print(f"  Test  : {len(X_test):>10,}  ({len(X_test)/total*100:.1f} %)")

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, le


# ==============================================================
# STEP 9 -- FEATURE SCALING
# Scaler is fitted ONLY on training data to prevent leakage,
# then applied identically to val and test.
# ==============================================================
def scale_features(X_train, X_val, X_test, feature_cols):
    print("\n" + "=" * 60)
    print("STEP 9: FEATURE SCALING")
    print("=" * 60)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    joblib.dump(scaler, MODELS_DIR / "scaler_34.pkl")
    print(f"Scaler saved  ->  {MODELS_DIR / 'scaler_34.pkl'}")
    print("StandardScaler fitted on training data only.")

    # Wrap back into DataFrames (preserves column names for downstream use)
    X_train_s = pd.DataFrame(X_train_s, columns=feature_cols, dtype=np.float32)
    X_val_s   = pd.DataFrame(X_val_s,   columns=feature_cols, dtype=np.float32)
    X_test_s  = pd.DataFrame(X_test_s,  columns=feature_cols, dtype=np.float32)

    return X_train_s, X_val_s, X_test_s, scaler


# ==============================================================
# STEP 10 -- SAVE PROCESSED DATA
# All splits saved as Parquet for fast I/O during training.
# ==============================================================
def save_data(X_train, X_val, X_test, y_train, y_val, y_test):
    print("\n" + "=" * 60)
    print("STEP 10: SAVING PROCESSED DATA")
    print("=" * 60)

    OUTPUT_DIR.mkdir(exist_ok=True)

    X_train.to_parquet(OUTPUT_DIR / "X_train.parquet", index=False)
    X_val.to_parquet(  OUTPUT_DIR / "X_val.parquet",   index=False)
    X_test.to_parquet( OUTPUT_DIR / "X_test.parquet",  index=False)

    pd.DataFrame({"Label": y_train}).to_parquet(OUTPUT_DIR / "y_train.parquet", index=False)
    pd.DataFrame({"Label": y_val  }).to_parquet(OUTPUT_DIR / "y_val.parquet",   index=False)
    pd.DataFrame({"Label": y_test }).to_parquet(OUTPUT_DIR / "y_test.parquet",  index=False)

    print(f"Saved to: {OUTPUT_DIR}/")
    for name in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
        size_mb = (OUTPUT_DIR / f"{name}.parquet").stat().st_size / 1024 ** 2
        print(f"  {name}.parquet   {size_mb:.1f} MB")


# ==============================================================
# STEP 11 -- SAVE METADATA
# JSON file consumed by the future training script to avoid
# hard-coded constants.
# ==============================================================
def save_metadata(feature_cols, le, n_train, n_val, n_test):
    print("\n" + "=" * 60)
    print("STEP 11: SAVING METADATA")
    print("=" * 60)

    metadata = {
        "num_features"  : len(feature_cols),
        "num_classes"   : int(len(le.classes_)),
        "feature_names" : list(feature_cols),
        "class_names"   : list(le.classes_),
        "train_samples" : int(n_train),
        "val_samples"   : int(n_val),
        "test_samples"  : int(n_test),
        "scaler_path"   : str(MODELS_DIR / "scaler_34.pkl"),
        "encoder_path"  : str(MODELS_DIR / "label_encoder_34.pkl"),
    }

    metadata_path = BASE_DIR / "data" / "metadata_34.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"{metadata_path} saved")
    print(f"  num_features : {metadata['num_features']}")
    print(f"  num_classes  : {metadata['num_classes']}")
    print(f"  class_names  : {metadata['class_names']}")


# ==============================================================
# STEP 12 -- FINAL VALIDATION CHECKS
# Mandatory checks before reporting success.
# ==============================================================
def final_checks(X_train, X_val, X_test, y_train, y_val, y_test, le, final_size):
    print("\n" + "=" * 60)
    print("STEP 12: FINAL VALIDATION CHECKS")
    print("=" * 60)

    passed = True

    # 1. No NaN in any split
    for name, X in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        n_nan = int(X.isnull().sum().sum())
        tag   = "PASS" if n_nan == 0 else "FAIL"
        if n_nan > 0:
            passed = False
        print(f"  [{tag}] {name}: {n_nan} NaN values")

    # 2. Dataset size close to TARGET_TOTAL (within ?15 %)
    total = len(y_train) + len(y_val) + len(y_test)
    ratio = total / TARGET_TOTAL
    tag   = "PASS" if 0.85 <= ratio <= 1.15 else "WARN"
    print(f"  [{tag}] Total samples: {total:,}  (target {TARGET_TOTAL:,}, ratio {ratio:.2f})")

    # 3. All classes present after sampling
    n_cls       = len(le.classes_)
    all_classes = set(range(n_cls))
    for split_name, y_split in [("train", y_train), ("val", y_val), ("test", y_test)]:
        present = set(np.unique(y_split))
        missing = all_classes - present
        tag = "PASS" if not missing else "WARN"
        missing_names = [le.classes_[i] for i in sorted(missing)]
        detail = "" if not missing else f"  missing: {missing_names}"
        print(f"  [{tag}] {split_name} split: {len(present)}/{n_cls} classes present{detail}")

    # 4. Stratification: max per-class proportion difference train vs val ? 2 %
    train_dist = pd.Series(y_train).value_counts(normalize=True)
    val_dist   = pd.Series(y_val).value_counts(normalize=True)
    max_diff   = (train_dist - val_dist).abs().max()
    tag        = "PASS" if max_diff <= 0.02 else "WARN"
    print(f"  [{tag}] Stratification max-diff train vs val: {max_diff:.4f}")

    status = "ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED -- see above"
    print(f"\n  Result: {status}")
    return passed


# ==============================================================
# MAIN
# ==============================================================
def main():
    print("=" * 60)
    print("34-CLASS PREPROCESSING PIPELINE")
    print("=" * 60)

    # Step 1 -- Load
    df = load_all_data(DATA_DIR)
    original_size = len(df)
    print(f"\nOriginal dataset size: {original_size:,} rows")

    # Step 2 -- Clean
    df = clean_data(df)

    # Step 3 -- Optimise dtypes
    df = optimize_dtypes(df)

    # Step 4 -- Distribution before balancing
    print("\n" + "=" * 60)
    print("STEP 4: CLASS DISTRIBUTION ANALYSIS")
    print("=" * 60)
    print_class_distribution(df, "Before balancing:")

    # Steps 5-6 -- Balance + sample
    df = smart_balance(df)

    # Distribution after balancing
    print_class_distribution(df, "After balancing:")
    final_size = len(df)

    # Step 7 -- Shuffle
    df = shuffle_data(df)

    # Step 8 -- Split
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, le = split_data(df)

    # Step 9 -- Scale
    X_train, X_val, X_test, _ = scale_features(X_train, X_val, X_test, feature_cols)

    # Step 10 -- Save splits
    save_data(X_train, X_val, X_test, y_train, y_val, y_test)

    # Step 11 -- Save metadata
    save_metadata(feature_cols, le, len(y_train), len(y_val), len(y_test))

    # Step 12 -- Validate
    final_checks(X_train, X_val, X_test, y_train, y_val, y_test, le, final_size)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print(f"  Original size : {original_size:,} rows")
    print(f"  Final size    : {final_size:,} rows")
    print(f"  Features      : {len(feature_cols)}")
    print(f"  Classes       : {len(le.classes_)}")
    print(f"  Outputs       : data/processed_data_34/")
    print(f"  Models        : models/preprocessing/")
    print("=" * 60)
    print("\nNext step: implement train-34.py")


if __name__ == "__main__":
    main()
