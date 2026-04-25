"""34-class logistic regression training. Run from '34 classes/' directory."""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

DATA_DIR         = Path("data/processed_data_34")
MODELS_PREP      = Path("models/preprocessing")
MODELS_TRAINED   = Path("models/trained")
RESULTS_DIR      = Path("results/logistic_34")
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
    print(f"y_train : {y_train.shape}  |  classes: {np.unique(y_train)}")

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

    configs = [
        {"solver": "lbfgs",  "max_iter": 1000},
        {"solver": "lbfgs",  "max_iter": 2000},
        {"solver": "saga",   "max_iter": 2000},
    ]

    model = None
    for cfg in configs:
        print(f"Trying solver={cfg['solver']}, max_iter={cfg['max_iter']} ...")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            candidate = LogisticRegression(
                solver=cfg["solver"],
                max_iter=cfg["max_iter"],
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            )
            candidate.fit(X_train, y_train)
            converged = not any(
                issubclass(w.category, Warning) and "ConvergenceWarning" in str(w.category)
                for w in caught
            )

        if converged:
            print(f"Converged with solver={cfg['solver']}, max_iter={cfg['max_iter']}")
            model = candidate
            break
        else:
            print(f"  Did not converge, retrying ...")

    if model is None:
        print("WARNING: Did not fully converge. Using last attempted configuration.")
        model = candidate

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

    cm = confusion_matrix(y_test, y_pred_test)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.3,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title("Logistic Regression -- 34-Class Confusion Matrix (Normalised)", fontsize=13)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()

    out_path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    print(f"Confusion matrix saved -> {out_path}")
    plt.show()


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


def save_model(model):
    print("\n" + "=" * 60)
    print("STEP 10: SAVING MODEL")
    print("=" * 60)

    out_path = MODELS_TRAINED / "logistic_34.pkl"
    joblib.dump(model, out_path)
    print(f"Model saved -> {out_path}")


def main():
    print("=" * 60)
    print("34-CLASS LOGISTIC REGRESSION TRAINING")
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
    save_model(model)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  Model : models/trained/logistic_34.pkl")
    print(f"  Plot  : results/logistic_34/confusion_matrix.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
