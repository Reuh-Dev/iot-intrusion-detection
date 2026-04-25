"""
Generate standardized result images for 2_8_CLASSES pipeline.

Output per results folder (logistic_2_8 / rf_2_8):
  confusion_matrix_binary.png
  confusion_matrix_8class.png
  metrics_table_binary.png
  metrics_table_8class.png
  per_class_metrics_table_binary.png
  per_class_metrics_table_8class.png

Run from: 2_8_CLASSES/
  python results/generate_results_2_8.py
"""

import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
)

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "DejaVu Sans"

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data/processed_data_2_8")
MODELS_DIR = Path("models/trained")

BINARY_CLASSES = ["Attack", "Benign"]
MULTI_CLASSES  = ["Benign", "BruteForce", "DDoS", "DoS",
                  "Mirai", "Recon", "Spoofing", "Web"]

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data splits...")
X_train  = pd.read_parquet(DATA_DIR / "X_train.parquet")
X_val    = pd.read_parquet(DATA_DIR / "X_val.parquet")
X_test   = pd.read_parquet(DATA_DIR / "X_test.parquet")
y2_train = pd.read_parquet(DATA_DIR / "y2_train.parquet")["Label_2"]
y2_val   = pd.read_parquet(DATA_DIR / "y2_val.parquet")["Label_2"]
y2_test  = pd.read_parquet(DATA_DIR / "y2_test.parquet")["Label_2"]
y8_train = pd.read_parquet(DATA_DIR / "y8_train.parquet")["Label_8"]
y8_val   = pd.read_parquet(DATA_DIR / "y8_val.parquet")["Label_8"]
y8_test  = pd.read_parquet(DATA_DIR / "y8_test.parquet")["Label_8"]
print(f"  Train {len(X_train):,}  Val {len(X_val):,}  Test {len(X_test):,}")


# ══════════════════════════════════════════════════════════════════════════
# 1 — Confusion Matrix  (test set, normalised %)
# ══════════════════════════════════════════════════════════════════════════
def save_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    cm     = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    n      = len(class_names)
    sz     = max(7, n * 1.1)

    annot  = np.where(cm_pct < 0.5, "", cm_pct.round(1).astype(str))

    fig, ax = plt.subplots(figsize=(sz, sz * 0.88))
    sns.heatmap(
        cm_pct, ax=ax, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.4, linecolor="#cccccc",
        cbar_kws={"label": "Percentage (%)"},
        vmin=0, vmax=100,
        annot=annot, fmt="",
        annot_kws={"size": 7 if n > 8 else 10},
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14)
    ax.set_ylabel("True Label",      fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.tick_params(axis="x", rotation=45, labelsize=8 if n > 8 else 10)
    ax.tick_params(axis="y", rotation=0,  labelsize=8 if n > 8 else 10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════
# 2 — Macro Metrics Table  (splits = list of (name, y_true, y_pred))
# ══════════════════════════════════════════════════════════════════════════
def save_metrics_table(splits, title, save_path):
    rows = []
    for name, y_true, y_pred in splits:
        rows.append({
            "Split":     name,
            "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
            "Precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
            "Recall":    round(recall_score(y_true, y_pred,    average="macro", zero_division=0), 4),
            "F1":        round(f1_score(y_true, y_pred,        average="macro", zero_division=0), 4),
        })
    df     = pd.DataFrame(rows)
    cols   = list(df.columns)
    n_rows = len(df)
    n_cols = len(cols)

    fig, ax = plt.subplots(figsize=(10, 0.7 + n_rows * 0.6))
    ax.axis("off")
    fig.patch.set_facecolor("white")

    tbl = ax.table(
        cellText=df.values, colLabels=cols,
        cellLoc="center", loc="center",
        colWidths=[0.18] + [0.205] * (n_cols - 1),
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)

    for j in range(n_cols):
        c = tbl[0, j]
        c.set_facecolor("#2c3e50")
        c.set_text_props(color="white", fontweight="bold")
        c.set_edgecolor("#bdc3c7")
        c.set_height(0.22)

    for i in range(1, n_rows + 1):
        bg = "#f2f4f5" if i % 2 == 0 else "white"
        for j in range(n_cols):
            c = tbl[i, j]
            c.set_facecolor(bg)
            c.set_edgecolor("#bdc3c7")
            c.set_height(0.22)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=14)
    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════
# 3 — Per-Class Metrics Table  (test set)
# ══════════════════════════════════════════════════════════════════════════
def save_per_class_table(y_true, y_pred, class_names, title, save_path):
    rep  = classification_report(
        y_true, y_pred, labels=class_names,
        target_names=class_names, output_dict=True, zero_division=0,
    )
    rows = [{"Class": c,
             "Precision": round(rep[c]["precision"], 4),
             "Recall":    round(rep[c]["recall"],    4),
             "F1-Score":  round(rep[c]["f1-score"],  4),
             "Support":   int(rep[c]["support"])}
            for c in class_names]
    df     = pd.DataFrame(rows)
    cols   = list(df.columns)
    n_rows = len(df)
    n_cols = len(cols)
    fig_h  = 0.55 + n_rows * 0.42 + 1.0

    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.axis("off")
    fig.patch.set_facecolor("white")

    tbl = ax.table(
        cellText=df.values, colLabels=cols,
        cellLoc="center", loc="lower center",
        colWidths=[0.28] + [0.18] * (n_cols - 1),
        bbox=[0, 0, 1, 0.94],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    for j in range(n_cols):
        c = tbl[0, j]
        c.set_facecolor("#2c3e50")
        c.set_text_props(color="white", fontweight="bold")
        c.set_edgecolor("#bdc3c7")
        c.set_height(0.055)

    for i in range(1, n_rows + 1):
        bg = "#f2f4f5" if i % 2 == 0 else "white"
        for j in range(n_cols):
            c = tbl[i, j]
            c.set_facecolor(bg)
            c.set_edgecolor("#bdc3c7")
            c.set_height(0.055)
            c.set_text_props(ha="left" if j == 0 else "center")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10, y=0.995)
    plt.tight_layout(pad=0.4)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════════
# MODELS CONFIG
# use_full_train=True  → predict on full 3.4M train set (slow, for logistic)
# use_full_train=False → show Test row only in metrics table (fast, for RF)
# ══════════════════════════════════════════════════════════════════════════
MODELS = [
    {
        "label":          "Logistic Regression",
        "binary_file":    "logreg_binary_2_8.pkl",
        "multi_file":     "logreg_multiclass_2_8.pkl",
        "results_dir":    Path("results/logistic_2_8"),
        "use_full_train": True,
    },
    {
        "label":          "Random Forest",
        "binary_file":    "binary_rf_2_8.pkl",
        "multi_file":     "multiclass_rf_2_8.pkl",
        "results_dir":    Path("results/rf_2_8"),
        "use_full_train": True,
    },
]

for cfg in MODELS:
    label       = cfg["label"]
    out         = cfg["results_dir"]
    full_train  = cfg["use_full_train"]
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {label.upper()}")
    print(f"{'='*60}")

    # ── Binary model ───────────────────────────────────────────────────────
    print(f"\n  Loading binary model...")
    gc.collect()
    bm = joblib.load(MODELS_DIR / cfg["binary_file"])

    b_te = bm.predict(X_test)
    if full_train:
        b_tr = bm.predict(X_train)
        b_va = bm.predict(X_val)
        bin_splits = [("Train", y2_train, b_tr),
                      ("Validation", y2_val, b_va),
                      ("Test", y2_test, b_te)]
    else:
        bin_splits = [("Test", y2_test, b_te)]

    del bm; gc.collect()

    print("  [Binary]")
    save_confusion_matrix(
        y2_test, b_te, BINARY_CLASSES,
        f"{label} -- Binary Confusion Matrix (%)",
        out / "confusion_matrix_binary.png",
    )
    save_metrics_table(
        bin_splits,
        f"{label} -- Binary Metrics (Macro)",
        out / "metrics_table_binary.png",
    )
    save_per_class_table(
        y2_test, b_te, BINARY_CLASSES,
        f"{label} -- Binary Per-Class Metrics (Test Set)",
        out / "per_class_metrics_table_binary.png",
    )

    # ── 8-class model ──────────────────────────────────────────────────────
    print(f"\n  Loading 8-class model...")
    gc.collect()
    mm = joblib.load(MODELS_DIR / cfg["multi_file"])

    m_te = mm.predict(X_test)
    if full_train:
        m_tr = mm.predict(X_train)
        m_va = mm.predict(X_val)
        multi_splits = [("Train", y8_train, m_tr),
                        ("Validation", y8_val, m_va),
                        ("Test", y8_test, m_te)]
    else:
        multi_splits = [("Test", y8_test, m_te)]

    del mm; gc.collect()

    print("  [8-Class]")
    save_confusion_matrix(
        y8_test, m_te, MULTI_CLASSES,
        f"{label} -- 8-Class Confusion Matrix (%)",
        out / "confusion_matrix_8class.png",
    )
    save_metrics_table(
        multi_splits,
        f"{label} -- 8-Class Metrics (Macro)",
        out / "metrics_table_8class.png",
    )
    save_per_class_table(
        y8_test, m_te, MULTI_CLASSES,
        f"{label} -- 8-Class Per-Class Metrics (Test Set)",
        out / "per_class_metrics_table_8class.png",
    )

print("\nAll done.")
