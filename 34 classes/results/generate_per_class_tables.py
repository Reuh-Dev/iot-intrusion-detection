"""
Generate per-class metrics tables for logistic_34 and rf_34.
Run from: 34 classes/
  python generate_per_class_tables.py
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data/processed_data_34")
MODELS_DIR = Path("models/trained")

# ── Load data (test split only) ───────────────────────────────────────────
print("Loading test data...")
X_test  = pd.read_parquet(DATA_DIR / "X_test.parquet")
y_enc   = pd.read_parquet(DATA_DIR / "y_test.parquet").squeeze()
le      = joblib.load(Path("models/preprocessing/label_encoder_34.pkl"))
y_test  = pd.Series(le.inverse_transform(y_enc), name="label")
print(f"  Test samples: {len(X_test):,}  |  Features: {X_test.shape[1]}")

class_names = list(le.classes_)
N = len(y_test)

# ── Helper: build per-class metrics dataframe ─────────────────────────────
def build_metrics_df(y_true, y_pred, class_names):
    report = classification_report(
        y_true, y_pred,
        labels=class_names,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for cls in class_names:
        r = report[cls]
        rows.append({
            "Class":     cls,
            "Precision": round(r["precision"], 4),
            "Recall":    round(r["recall"],    4),
            "F1-Score":  round(r["f1-score"],  4),
            "Support":   int(r["support"]),
        })
    return pd.DataFrame(rows)

# ── Helper: render and save table ─────────────────────────────────────────
def save_table(df, title, save_path):
    cols        = list(df.columns)
    n_rows      = len(df)
    n_cols      = len(cols)
    row_h       = 0.38
    header_h    = 0.55
    fig_h       = header_h + n_rows * row_h + 1.2   # extra for title + footer
    fig_w       = 14

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Column widths (class name wider)
    col_widths = [0.30] + [0.14] * (n_cols - 1)

    tbl = ax.table(
        cellText=df.values,
        colLabels=cols,
        cellLoc="center",
        loc="lower center",
        colWidths=col_widths,
        bbox=[0, 0, 1, 0.94],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)

    # Style header row
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#bdc3c7")
        cell.set_height(0.055)

    # Style data rows (alternating)
    for i in range(1, n_rows + 1):
        bg = "#f2f4f5" if i % 2 == 0 else "white"
        for j in range(n_cols):
            cell = tbl[i, j]
            cell.set_facecolor(bg)
            cell.set_edgecolor("#bdc3c7")
            cell.set_height(0.038)
            cell.set_text_props(ha="left" if j == 0 else "center")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10, y=0.995)

    plt.tight_layout(pad=0.4)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")

# ── Process each model ─────────────────────────────────────────────────────
for model_file, label, results_dir in [
    ("logistic_34.pkl", "Logistic Regression", Path("results/logistic_34")),
    ("rf_34.pkl",       "Random Forest",       Path("results/rf_34")),
]:
    print(f"\nEvaluating {label}...")
    model  = joblib.load(MODELS_DIR / model_file)
    y_pred = pd.Series(le.inverse_transform(model.predict(X_test)))

    df = build_metrics_df(y_test, y_pred, class_names)

    title     = f"{label} -- Per-Class Metrics (Test Set, 34 Classes)"
    save_path = results_dir / "per_class_metrics_table.png"
    save_table(df, title, save_path)

print("\nDone.")
