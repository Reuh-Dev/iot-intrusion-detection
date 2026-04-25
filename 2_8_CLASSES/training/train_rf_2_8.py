"""Random Forest training for CICIoT2023 (2/8-class pipeline). Run from 2_8_CLASSES/ directory."""

import json
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from tqdm import tqdm
import joblib
import warnings
warnings.filterwarnings('ignore')


class Config:
    # Paths (run from 2_8_CLASSES/)
    DATA_DIR    = Path("data/processed_data_2_8")
    MODELS_DIR  = Path("models/trained")
    RESULTS_DIR = Path("results/rf_2_8")

    N_CPUS = os.cpu_count() or 4
    N_JOBS = max(1, N_CPUS - 1)   # Leave one core free for OS

    RANDOM_STATE    = 42
    CV_FOLDS        = 3
    N_ITER_SEARCH   = 12

    USE_SMOTE        = False
    TUNE_HYPERPARAMS = True
    TUNE_SUBSET_SIZE = 150_000      # Safe for 8-16 GB RAM

    MODEL_TYPE = "random_forest"

    PLOT_CONFUSION_MATRICES = False
    PLOT_FEATURE_IMPORTANCE = False
    SAVE_LOGS               = False

    @classmethod
    def setup_directories(cls):
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def print_config(cls):
        print(f"Hardware: {cls.N_CPUS} CPU cores, using {cls.N_JOBS} parallel jobs")
        print(f"Model: {cls.MODEL_TYPE.upper()}")
        print(f"Tuning subset: {cls.TUNE_SUBSET_SIZE:,} samples")
        print(f"CV folds: {cls.CV_FOLDS} | Parameter combos: {cls.N_ITER_SEARCH}")


def setup_logging():
    logger = logging.getLogger("Training_RF_2_8")
    logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    console.setFormatter(formatter)

    if Config.SAVE_LOGS:
        Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        log_file = Config.RESULTS_DIR / f"training_{datetime.now():%Y%m%d_%H%M%S}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

    logger.addHandler(console)
    return logger

logger = setup_logging()


def load_data():
    logger.info("Loading preprocessed Parquet files...")

    X_train = pd.read_parquet(Config.DATA_DIR / "X_train.parquet")
    X_val   = pd.read_parquet(Config.DATA_DIR / "X_val.parquet")
    X_test  = pd.read_parquet(Config.DATA_DIR / "X_test.parquet")

    y8_train = pd.read_parquet(Config.DATA_DIR / "y8_train.parquet")['Label_8']
    y8_val   = pd.read_parquet(Config.DATA_DIR / "y8_val.parquet")['Label_8']
    y8_test  = pd.read_parquet(Config.DATA_DIR / "y8_test.parquet")['Label_8']

    y2_train = pd.read_parquet(Config.DATA_DIR / "y2_train.parquet")['Label_2']
    y2_val   = pd.read_parquet(Config.DATA_DIR / "y2_val.parquet")['Label_2']
    y2_test  = pd.read_parquet(Config.DATA_DIR / "y2_test.parquet")['Label_2']

    logger.info(f"Train: {len(X_train):,} samples, {X_train.shape[1]} features")
    logger.info(f"Val:   {len(X_val):,} samples")
    logger.info(f"Test:  {len(X_test):,} samples")

    return X_train, X_val, X_test, y8_train, y8_val, y8_test, y2_train, y2_val, y2_test


def get_base_model():
    return RandomForestClassifier(
        random_state=Config.RANDOM_STATE,
        n_jobs=Config.N_JOBS,
        class_weight='balanced',
        verbose=0
    )

def get_param_grid():
    return {
        'n_estimators':      [100, 200, 300],
        'max_depth':         [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf':  [1, 2, 4],
    }


def tune_on_subset(X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
    logger.info(f"Selecting {Config.TUNE_SUBSET_SIZE:,} stratified samples for tuning...")

    X_sub, _, y_sub, _ = train_test_split(
        X_train, y_train,
        train_size=Config.TUNE_SUBSET_SIZE,
        stratify=y_train,
        random_state=Config.RANDOM_STATE
    )

    base_model = get_base_model()
    param_dist = get_param_grid()

    cv = StratifiedKFold(n_splits=Config.CV_FOLDS, shuffle=True, random_state=Config.RANDOM_STATE)

    search = RandomizedSearchCV(
        base_model, param_dist,
        n_iter=Config.N_ITER_SEARCH,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=Config.N_JOBS,
        verbose=0,
        random_state=Config.RANDOM_STATE
    )

    logger.info(f"Starting RandomizedSearchCV with {Config.N_ITER_SEARCH} combos x {Config.CV_FOLDS} folds = {Config.N_ITER_SEARCH * Config.CV_FOLDS} fits")

    with tqdm(total=Config.N_ITER_SEARCH * Config.CV_FOLDS, desc="Tuning hyperparameters", unit="fit") as pbar:
        original_fit = search.fit
        def fit_with_progress(*args, **kwargs):
            result = original_fit(*args, **kwargs)
            pbar.update(Config.N_ITER_SEARCH * Config.CV_FOLDS - pbar.n)
            return result
        search.fit = fit_with_progress
        search.fit(X_sub, y_sub)

    logger.info(f"Best parameters: {search.best_params_}")
    logger.info(f"Best CV score (weighted F1): {search.best_score_:.4f}")
    return search.best_params_


def train_final_model(X_train: pd.DataFrame, y_train: pd.Series, best_params: Dict[str, Any]):
    logger.info("Training final model on full dataset...")

    model = get_base_model()
    model.set_params(**best_params)

    logger.info(f"Building {model.n_estimators} trees using {Config.N_JOBS} cores...")
    model.fit(X_train, y_train)
    logger.info("Training complete.")
    return model


def evaluate_model(model, X, y, model_name, dataset_name):
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, output_dict=True)

    logger.info(f"\n{model_name} - {dataset_name} Results:")
    logger.info("\n" + classification_report(y, y_pred, digits=4))

    if Config.PLOT_CONFUSION_MATRICES:
        plot_confusion_matrix(y, y_pred, model_name, dataset_name)

    return {
        'accuracy':    report['accuracy'],
        'macro_f1':    report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
    }

def plot_confusion_matrix(y_true, y_pred, model_name, dataset_name):
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=sorted(y_true.unique()),
                yticklabels=sorted(y_true.unique()))
    plt.title(f'{model_name} - {dataset_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    filename = Config.RESULTS_DIR / f"confusion_matrix_{model_name}_{dataset_name}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix to {filename}")

def plot_feature_importance(model, feature_names, model_name):
    if not hasattr(model, 'feature_importances_'):
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(indices)), importances[indices][::-1])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices[::-1]])
    plt.xlabel('Feature Importance')
    plt.title(f'{model_name} - Top 20 Features')
    plt.tight_layout()
    filename = Config.RESULTS_DIR / f"feature_importance_{model_name}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved feature importance plot to {filename}")


def save_model(model, model_name, metadata):
    model_path = Config.MODELS_DIR / f"{model_name}.pkl"
    joblib.dump(model, model_path)
    if Config.SAVE_LOGS:
        metadata_path = Config.MODELS_DIR / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Model saved to {model_path}")


def main():
    print("\n" + "=" * 80)
    print("RANDOM FOREST TRAINING  [2/8-CLASS PIPELINE]  CICIoT2023")
    Config.print_config()
    print("=" * 80 + "\n")

    Config.setup_directories()

    X_train, X_val, X_test, y8_train, y8_val, y8_test, y2_train, y2_val, y2_test = load_data()

    logger.info("\n" + "=" * 60)
    logger.info("BINARY CLASSIFICATION (Attack vs Benign)")
    logger.info("=" * 60)

    best_params_bin = tune_on_subset(X_train, y2_train) if Config.TUNE_HYPERPARAMS else {}
    binary_model = train_final_model(X_train, y2_train, best_params_bin)

    bin_results = {}
    for name, X, y in [('train', X_train, y2_train), ('val', X_val, y2_val), ('test', X_test, y2_test)]:
        bin_results[name] = evaluate_model(binary_model, X, y, 'Binary', name)

    save_model(binary_model, "binary_rf_2_8", {
        'type':        'binary',
        'best_params': best_params_bin,
        'features':    X_train.columns.tolist(),
        'results':     bin_results
    })

    logger.info("\n" + "=" * 60)
    logger.info("8-CLASS CLASSIFICATION")
    logger.info("=" * 60)

    best_params_multi = tune_on_subset(X_train, y8_train) if Config.TUNE_HYPERPARAMS else best_params_bin
    multiclass_model = train_final_model(X_train, y8_train, best_params_multi)

    multi_results = {}
    for name, X, y in [('train', X_train, y8_train), ('val', X_val, y8_val), ('test', X_test, y8_test)]:
        multi_results[name] = evaluate_model(multiclass_model, X, y, '8-Class', name)

    if Config.PLOT_FEATURE_IMPORTANCE:
        plot_feature_importance(multiclass_model, X_train.columns.tolist(), '8-Class')

    save_model(multiclass_model, "multiclass_rf_2_8", {
        'type':        'multiclass',
        'num_classes': 8,
        'best_params': best_params_multi,
        'features':    X_train.columns.tolist(),
        'results':     multi_results
    })

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info(f"Binary  - Test Acc: {bin_results['test']['accuracy']:.4f}  |  Weighted F1: {bin_results['test']['weighted_f1']:.4f}")
    logger.info(f"8-Class - Test Acc: {multi_results['test']['accuracy']:.4f}  |  Weighted F1: {multi_results['test']['weighted_f1']:.4f}")
    logger.info(f"8-Class - Macro F1: {multi_results['test']['macro_f1']:.4f}")
    logger.info(f"\nModels:  {Config.MODELS_DIR}/")
    logger.info(f"Results: {Config.RESULTS_DIR}/")
    print("\nAll done!\n")

if __name__ == "__main__":
    main()
