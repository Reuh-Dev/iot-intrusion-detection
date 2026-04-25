"""34-class random forest training. Run from '34 classes/' directory."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

DATA_DIR       = Path("data/processed_data_34")
MODELS_PREP    = Path("models/preprocessing")
MODELS_TRAINED = Path("models/trained")
RESULTS_DIR    = Path("results/rf_34")
MODELS_TRAINED.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SAVE_PLOTS = False


def load_data():
    print("=" * 60)
    print("STEP 1: LOADING DATA")
    print("=" * 60)

    X_train = pd.read_parquet(DATA_DIR / "X_train.parquet")
    X_val   = pd.read_parquet(DATA_DIR / "X_val.parquet")
    X_test  = pd.read_parquet(DATA_DIR / "X_test.parquet")

    y_train = pd.read_parquet(DATA_DIR / "y_train.parquet")["Label"].values
    y_val   = pd.read_parquet(DATA_DIR / "y_val.parquet")["Label"].values
    y_test  = pd.read_parquet(DATA_DIR / "y_test.parquet")["Label"].values

    print(f"X_train : {X_train.shape}")
    print(f"X_val   : {X_val.shape}")
    print(f"X_test  : {X_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_encoder():
    print("\n" + "=" * 60)
    print("STEP 2: LOADING LABEL ENCODER")
    print("=" * 60)

    le = joblib.load(MODELS_PREP / "label_encoder_34.pkl")
    class_names = list(le.classes_)
    print(f"Classes ({len(class_names)}): {class_names}")
    return le, class_names


def train_model(X_train, y_train):
    print("\n" + "=" * 60)
    print("STEP 3-4: MODEL TRAINING")
    print("=" * 60)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    print("Training Random Forest (200 trees, all cores) ...")
    model.fit(X_train, y_train)
    print("Training complete.")
    return model


def predict_all(model, X_train, X_val, X_test):
    print("\n" + "=" * 60)
    print("STEP 5: GENERATING PREDICTIONS")
    print("=" * 60)

    y_pred_train = model.predict(X_train)
    y_pred_val   = model.predict(X_val)
    y_pred_test  = model.predict(X_test)

    print("Predictions generated for train, val, and test sets.")
    return y_pred_train, y_pred_val, y_pred_test


def evaluate(y_true, y_pred, split_name: str) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n=== {split_name} RESULTS ===")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}  (macro)")
    print(f"Recall    : {rec:.4f}  (macro)")
    print(f"F1        : {f1:.4f}  (macro)")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def print_classification_report(y_test, y_pred_test, class_names):
    print("\n" + "=" * 60)
    print("STEP 7: CLASSIFICATION REPORT (TEST SET)")
    print("=" * 60)
    print(classification_report(y_test, y_pred_test, target_names=class_names, zero_division=0))


def plot_confusion_matrix(y_test, y_pred_test, class_names):
    print("\n" + "=" * 60)
    print("STEP 8: CONFUSION MATRIX (TEST SET)")
    print("=" * 60)

    cm      = confusion_matrix(y_test, y_pred_test)
    cm_norm = (cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100).round(1)

    fig, ax = plt.subplots(figsize=(28, 24))
    sns.heatmap(
        cm_norm,
        ax=ax,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot=True,
        fmt=".1f",
        annot_kws={"size": 6.5},
        linewidths=0.3,
        linecolor="#cccccc",
        vmin=0, vmax=100,
        cbar_kws={"label": "Percentage (%)", "shrink": 0.6},
    )
    ax.set_xlabel("Predicted Label", fontsize=13, labelpad=14)
    ax.set_ylabel("True Label", fontsize=13, labelpad=14)
    ax.set_title("Random Forest -- 34-Class Confusion Matrix (%)", fontsize=15, pad=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8.5)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8.5)

    plt.tight_layout()
    out = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved -> {out}")


def overfitting_check(train_metrics, val_metrics, test_metrics):
    print("\n" + "=" * 60)
    print("STEP 9: OVERFITTING CHECK")
    print("=" * 60)

    print(f"{'Metric':<12} {'Train':>8} {'Val':>8} {'Test':>8}  {'Train-Val gap':>14}")
    print("-" * 55)
    for metric in ["accuracy", "precision", "recall", "f1"]:
        tr  = train_metrics[metric]
        val = val_metrics[metric]
        te  = test_metrics[metric]
        gap = tr - val
        print(f"{metric:<12} {tr:>8.4f} {val:>8.4f} {te:>8.4f}  {gap:>+14.4f}")

    gap_f1 = train_metrics["f1"] - val_metrics["f1"]
    if gap_f1 > 0.05:
        print("\n[RESULT] Possible overfitting detected (train F1 - val F1 > 0.05)")
    elif gap_f1 < -0.05:
        print("\n[RESULT] Unusual: val F1 significantly higher than train F1")
    else:
        print("\n[RESULT] No significant overfitting detected")


def save_all(model, train_metrics, val_metrics, test_metrics):
    print("\n" + "=" * 60)
    print("STEP 10: SAVING MODEL & RESULTS")
    print("=" * 60)

    joblib.dump(model, MODELS_TRAINED / "rf_34.pkl")
    print(f"Model saved -> {MODELS_TRAINED / 'rf_34.pkl'}")

    rows = []
    for name, m in [("Train", train_metrics), ("Validation", val_metrics), ("Test", test_metrics)]:
        rows.append({
            "Split":     name,
            "Accuracy":  round(m["accuracy"],  4),
            "Precision": round(m["precision"], 4),
            "Recall":    round(m["recall"],    4),
            "F1":        round(m["f1"],        4),
        })

    df = pd.DataFrame(rows)
    if SAVE_PLOTS:
        df.to_csv(RESULTS_DIR / "metrics.csv", index=False)
        print(f"Metrics CSV saved -> {RESULTS_DIR / 'metrics.csv'}")

        fig, ax = plt.subplots(figsize=(8, 2))
        ax.axis("off")
        tbl = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc="center", loc="center"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1.2, 1.8)
        plt.title("Random Forest -- 34-Class Metrics (Macro)", fontsize=12, pad=10)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "metrics_table.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Metrics table saved -> {RESULTS_DIR / 'metrics_table.png'}")


def main():
    print("=" * 60)
    print("34-CLASS RANDOM FOREST TRAINING")
    print("=" * 60)

    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    le, class_names = load_encoder()
    model = train_model(X_train, y_train)
    y_pred_train, y_pred_val, y_pred_test = predict_all(model, X_train, X_val, X_test)

    print("\n" + "=" * 60)
    print("STEP 6: EVALUATION METRICS")
    print("=" * 60)
    train_metrics = evaluate(y_train, y_pred_train, "TRAIN")
    val_metrics   = evaluate(y_val,   y_pred_val,   "VALIDATION")
    test_metrics  = evaluate(y_test,  y_pred_test,  "TEST")

    print_classification_report(y_test, y_pred_test, class_names)
    if SAVE_PLOTS:
        plot_confusion_matrix(y_test, y_pred_test, class_names)
    overfitting_check(train_metrics, val_metrics, test_metrics)
    save_all(model, train_metrics, val_metrics, test_metrics)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  Model   : models/trained/rf_34.pkl")
    print(f"  Results : results/rf_34/")
    print("=" * 60)


if __name__ == "__main__":
    main()
