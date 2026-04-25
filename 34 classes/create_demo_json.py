"""
Demo JSON Generator
Loads X_test / y_test, randomly picks 20 samples, and saves:
  - demo_samples.json       <- input for the API (no labels)
  - demo_true_labels.json   <- ground truth to compare after prediction
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

DATA_DIR    = Path("data/processed_data_34")
MODELS_PREP = Path("models/preprocessing")
N_SAMPLES   = 20
RANDOM_SEED = 42

# Load test data
X_test = pd.read_parquet(DATA_DIR / "X_test.parquet")
y_test = pd.read_parquet(DATA_DIR / "y_test.parquet")["Label"].values

# Load encoder to convert integer labels back to class names
le = joblib.load(MODELS_PREP / "label_encoder_34.pkl")
class_names = le.inverse_transform(y_test)

# Random sample
rng     = np.random.default_rng(RANDOM_SEED)
indices = rng.choice(len(X_test), size=N_SAMPLES, replace=False)
indices = sorted(indices)

X_sample = X_test.iloc[indices]
y_sample = class_names[indices]

# Build demo_samples.json — plain list of feature dicts, NO labels
samples = []
for _, row in X_sample.iterrows():
    samples.append({col: round(float(val), 6) for col, val in row.items()})

with open("demo_samples.json", "w") as f:
    json.dump(samples, f, indent=2)

# Build demo_true_labels.json — index + true class name for comparison
true_labels = [
    {"index": i, "true_class": str(label)}
    for i, label in enumerate(y_sample)
]

with open("demo_true_labels.json", "w") as f:
    json.dump(true_labels, f, indent=2)

# Summary
print("=" * 50)
print("DEMO FILES GENERATED")
print("=" * 50)
print(f"  Samples      : {N_SAMPLES}")
print(f"  demo_samples.json       -> send to /predict-batch or /predict-file")
print(f"  demo_true_labels.json   -> compare with API predictions")
print()
print("Class distribution in sample:")
from collections import Counter
for cls, cnt in sorted(Counter(y_sample).items()):
    print(f"  {cls:<35} {cnt}")
print()
print("=" * 50)
print("HOW TO USE")
print("=" * 50)
print()
print("1. Start the API:")
print("   uvicorn api_34:app --reload")
print()
print("2. Open Swagger UI:")
print("   http://127.0.0.1:8000/docs")
print()
print("3. Upload demo_samples.json to POST /predict-file")
print("   OR paste contents into POST /predict-batch")
print()
print("4. Compare predictions with demo_true_labels.json")
print("=" * 50)
