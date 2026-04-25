"""Logistic regression training for CICIoT2023 (2/8-class pipeline). Run from 2_8_CLASSES/ directory."""

import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class Config:
    # Paths (run from 2_8_CLASSES/)
    DATA_DIR    = Path("data/processed_data_2_8")
    MODELS_DIR  = Path("models/trained")
    RESULTS_DIR = Path("results/logistic_2_8")
    LOG_DIR     = Path("logs")

    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_GPU    = torch.cuda.is_available()
    NUM_WORKERS = 4 if USE_GPU else 0
    PIN_MEMORY  = USE_GPU

    RANDOM_SEED          = 42
    BATCH_SIZE           = 4096 if USE_GPU else 1024
    LEARNING_RATE        = 0.01
    WEIGHT_DECAY         = 1e-4
    EPOCHS               = 100
    EARLY_STOP_PATIENCE  = 10
    LR_REDUCE_PATIENCE   = 5
    LR_REDUCE_FACTOR     = 0.5

    USE_CLASS_WEIGHTS = True

    PLOT_CONFUSION_MATRICES = False
    PLOT_LEARNING_CURVES    = False
    SAVE_PLOTS              = False
    SAVE_LOGS               = False

    @classmethod
    def setup_directories(cls):
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        if cls.SAVE_LOGS:
            cls.LOG_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def print_config(cls):
        print("\n" + "=" * 80)
        print("LOGISTIC REGRESSION [2/8-CLASS PIPELINE] - CONFIGURATION")
        print("=" * 80)
        print(f"Device: {cls.DEVICE} (GPU: {cls.USE_GPU})")
        if not cls.USE_GPU:
            print("  Note: GPU not available. Install with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print(f"Batch size: {cls.BATCH_SIZE}")
        print(f"Learning rate: {cls.LEARNING_RATE}, Weight decay: {cls.WEIGHT_DECAY}")
        print(f"Max epochs: {cls.EPOCHS}, Early stop patience: {cls.EARLY_STOP_PATIENCE}")
        print(f"Class weights: {'enabled' if cls.USE_CLASS_WEIGHTS else 'disabled'}")
        print("=" * 80 + "\n")

# Set random seeds for reproducibility
torch.manual_seed(Config.RANDOM_SEED)
np.random.seed(Config.RANDOM_SEED)
if Config.USE_GPU:
    torch.cuda.manual_seed_all(Config.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_name: str = "logistic_regression_2_8") -> logging.Logger:
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
    console.setFormatter(console_format)

    if Config.SAVE_LOGS:
        Config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = Config.LOG_DIR / f"{log_name}_{datetime.now():%Y%m%d_%H%M%S}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

    logger.addHandler(console)

    return logger

logger = setup_logging()


def load_preprocessed_data():
    logger.info("Loading preprocessed data from %s", Config.DATA_DIR)

    X_train = pd.read_parquet(Config.DATA_DIR / "X_train.parquet")
    X_val   = pd.read_parquet(Config.DATA_DIR / "X_val.parquet")
    X_test  = pd.read_parquet(Config.DATA_DIR / "X_test.parquet")

    y2_train = pd.read_parquet(Config.DATA_DIR / "y2_train.parquet")['Label_2']
    y2_val   = pd.read_parquet(Config.DATA_DIR / "y2_val.parquet")['Label_2']
    y2_test  = pd.read_parquet(Config.DATA_DIR / "y2_test.parquet")['Label_2']

    y8_train = pd.read_parquet(Config.DATA_DIR / "y8_train.parquet")['Label_8']
    y8_val   = pd.read_parquet(Config.DATA_DIR / "y8_val.parquet")['Label_8']
    y8_test  = pd.read_parquet(Config.DATA_DIR / "y8_test.parquet")['Label_8']

    metadata_path = Path("data") / "metadata_2_8.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    feature_names  = metadata['feature_columns']
    class_names_8  = metadata['class_names_8']
    class_names_2  = ['Attack', 'Benign']

    label2idx_2 = {'Attack': 0, 'Benign': 1}
    y2_train = y2_train.map(label2idx_2).values
    y2_val   = y2_val.map(label2idx_2).values
    y2_test  = y2_test.map(label2idx_2).values

    label2idx_8 = {name: idx for idx, name in enumerate(class_names_8)}
    y8_train = y8_train.map(label2idx_8).values
    y8_val   = y8_val.map(label2idx_8).values
    y8_test  = y8_test.map(label2idx_8).values

    logger.info("Train samples: %d, Val: %d, Test: %d", len(X_train), len(X_val), len(X_test))
    logger.info("Features: %d", len(feature_names))
    logger.info("8-class distribution in train: %s", dict(pd.Series(y8_train).value_counts().to_dict()))

    return (X_train, X_val, X_test,
            y2_train, y2_val, y2_test,
            y8_train, y8_val, y8_test,
            feature_names, class_names_2, class_names_8)


class TabularDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y: np.ndarray):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def compute_class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    class_counts = np.bincount(y, minlength=num_classes)
    total = len(y)
    weights = total / (num_classes * class_counts)
    weights = np.clip(weights, 0.1, 10.0)
    return torch.tensor(weights, dtype=torch.float32)

class EarlyStopping:
    def __init__(self, patience: int = 10, delta: float = 0.001):
        self.patience   = patience
        self.delta      = delta
        self.best_score = None
        self.counter    = 0
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_f1: float, model: nn.Module) -> bool:
        if self.best_score is None or val_f1 > self.best_score + self.delta:
            self.best_score = val_f1
            self.counter    = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return True

    def load_best(self, model: nn.Module):
        model.load_state_dict(self.best_state)

def train_epoch(model, loader, criterion, optimizer, device, desc="Training"):
    model.train()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    pbar = tqdm(loader, desc=desc, leave=False)
    for X_batch, y_batch in pbar:
        X_batch, y_batch = X_batch.to(device, non_blocking=Config.PIN_MEMORY), y_batch.to(device, non_blocking=Config.PIN_MEMORY)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss    = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, macro_f1

def validate_epoch(model, loader, criterion, device, desc="Validation"):
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in tqdm(loader, desc=desc, leave=False):
            X_batch, y_batch = X_batch.to(device, non_blocking=Config.PIN_MEMORY), y_batch.to(device, non_blocking=Config.PIN_MEMORY)
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, macro_f1

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, epochs, early_stop_patience, model_name):
    early_stopper = EarlyStopping(patience=early_stop_patience)
    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}

    for epoch in range(1, epochs + 1):
        logger.info(f"Epoch {epoch}/{epochs}")

        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device,
                                            desc=f"Train {epoch}")
        val_loss, val_f1 = validate_epoch(model, val_loader, criterion, device,
                                           desc=f"Val {epoch}")

        if scheduler:
            scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        logger.info(f"  Train Loss: {train_loss:.4f} | Train Macro F1: {train_f1:.4f}")
        logger.info(f"  Val Loss:   {val_loss:.4f} | Val Macro F1:   {val_f1:.4f}")

        early_stopper(val_f1, model)
        if early_stopper.early_stop:
            logger.info(f"Early stopping triggered after {epoch} epochs.")
            break

    early_stopper.load_best(model)
    logger.info(f"Best validation Macro F1: {early_stopper.best_score:.4f}")

    if Config.PLOT_LEARNING_CURVES:
        plot_learning_curves(history, model_name)

    return model, history

def plot_learning_curves(history: Dict, model_name: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], label='Train')
    ax1.plot(epochs, history['val_loss'],   label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Loss')
    ax1.legend()

    ax2.plot(epochs, history['train_f1'], label='Train')
    ax2.plot(epochs, history['val_f1'],   label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Macro F1')
    ax2.set_title(f'{model_name} - Macro F1')
    ax2.legend()

    plt.tight_layout()
    filename = Config.RESULTS_DIR / f"learning_curves_{model_name}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    logger.info(f"Learning curves saved to {filename}")


def evaluate_model(model, loader, device, class_names, model_name, dataset_name):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in tqdm(loader, desc=f"Evaluating {dataset_name}", leave=False):
            X_batch = X_batch.to(device, non_blocking=Config.PIN_MEMORY)
            outputs = model(X_batch)
            preds   = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())

    report = classification_report(all_labels, all_preds,
                                    target_names=class_names,
                                    output_dict=True, zero_division=0)

    logger.info(f"\n{model_name} - {dataset_name} Results:")
    logger.info("\n" + classification_report(all_labels, all_preds,
                                              target_names=class_names,
                                              zero_division=0))

    if Config.PLOT_CONFUSION_MATRICES:
        plot_confusion_matrix(all_labels, all_preds, class_names,
                              f"{model_name}_{dataset_name}")

    return {
        'accuracy':    report['accuracy'],
        'macro_f1':    report['macro avg']['f1-score'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'predictions': all_preds,
        'labels':      all_labels
    }

def plot_confusion_matrix(y_true, y_pred, class_names, title_suffix):
    plt.figure(figsize=(12, 10))
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Logistic Regression - {title_suffix}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    filename = Config.RESULTS_DIR / f"cm_logreg_{title_suffix}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to {filename}")

def plot_feature_importance(model, feature_names, class_names, model_name):
    weights = model.linear.weight.data.cpu().numpy()  # (num_classes, input_dim)
    if len(class_names) == 2:
        importance = np.abs(weights[1, :])
        title = f'{model_name} - Feature Importance (|Weight| for Benign)'
    else:
        importance = np.mean(np.abs(weights), axis=0)
        title = f'{model_name} - Feature Importance (Mean |Weight| across classes)'

    indices      = np.argsort(importance)[::-1][:20]
    top_features = [feature_names[i] for i in indices]
    top_importance = importance[indices]

    plt.figure(figsize=(12, 8))
    plt.barh(range(len(indices)), top_importance[::-1])
    plt.yticks(range(len(indices)), top_features[::-1])
    plt.xlabel('Importance (|Coefficient|)')
    plt.title(title)
    plt.tight_layout()
    filename = Config.RESULTS_DIR / f"feature_importance_{model_name}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Feature importance plot saved to {filename}")


def save_model(model, model_name, metadata):
    model_path = Config.MODELS_DIR / f"{model_name}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'metadata': metadata
    }, model_path)

    if Config.SAVE_LOGS:
        metadata_path = Config.MODELS_DIR / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Metadata saved to {metadata_path}")

    logger.info(f"Model saved to {model_path}")


def run_logistic_regression():
    Config.print_config()
    Config.setup_directories()

    (X_train, X_val, X_test,
     y2_train, y2_val, y2_test,
     y8_train, y8_val, y8_test,
     feature_names, class_names_2, class_names_8) = load_preprocessed_data()

    input_dim = len(feature_names)
    results   = {}

    logger.info("\n" + "=" * 60)
    logger.info("BINARY CLASSIFICATION (Attack vs Benign)")
    logger.info("=" * 60)

    train_dataset = TabularDataset(X_train, y2_train)
    val_dataset   = TabularDataset(X_val,   y2_val)
    test_dataset  = TabularDataset(X_test,  y2_test)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
                              num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
    val_loader   = DataLoader(val_dataset,   batch_size=Config.BATCH_SIZE, shuffle=False,
                              num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
    test_loader  = DataLoader(test_dataset,  batch_size=Config.BATCH_SIZE, shuffle=False,
                              num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)

    model_bin = LogisticRegressionModel(input_dim, num_classes=2).to(Config.DEVICE)

    if Config.USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(y2_train, num_classes=2).to(Config.DEVICE)
        logger.info(f"Class weights: {class_weights.cpu().numpy()}")
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model_bin.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

    try:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=Config.LR_REDUCE_FACTOR,
            patience=Config.LR_REDUCE_PATIENCE, verbose=True
        )
    except TypeError:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=Config.LR_REDUCE_FACTOR,
            patience=Config.LR_REDUCE_PATIENCE
        )
        logger.info("Using ReduceLROnPlateau without verbose (older PyTorch version)")

    model_bin, history_bin = train_model(model_bin, train_loader, val_loader,
                                          criterion, optimizer, scheduler,
                                          Config.DEVICE, Config.EPOCHS, Config.EARLY_STOP_PATIENCE,
                                          "Binary_LogReg_2_8")

    bin_results = {}
    for name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        bin_results[name] = evaluate_model(model_bin, loader, Config.DEVICE, class_names_2,
                                            "Binary Logistic Regression", name)

    if Config.SAVE_PLOTS:
        plot_feature_importance(model_bin, feature_names, class_names_2, "Binary_LogReg_2_8")

    save_model(model_bin, "logreg_binary_2_8", {
        'type':         'binary_logistic',
        'input_dim':    input_dim,
        'num_classes':  2,
        'class_names':  class_names_2,
        'feature_names': feature_names,
        'results': {k: {kk: vv for kk, vv in v.items() if kk not in ['predictions', 'labels']}
                    for k, v in bin_results.items()},
        'best_val_f1':  history_bin['val_f1'][-1] if history_bin['val_f1'] else None,
        'class_weights': class_weights.cpu().tolist() if class_weights is not None else None
    })

    results['binary'] = bin_results

    logger.info("\n" + "=" * 60)
    logger.info("8-CLASS CLASSIFICATION")
    logger.info("=" * 60)

    train_dataset8 = TabularDataset(X_train, y8_train)
    val_dataset8   = TabularDataset(X_val,   y8_val)
    test_dataset8  = TabularDataset(X_test,  y8_test)

    train_loader8 = DataLoader(train_dataset8, batch_size=Config.BATCH_SIZE, shuffle=True,
                                num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
    val_loader8   = DataLoader(val_dataset8,   batch_size=Config.BATCH_SIZE, shuffle=False,
                                num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)
    test_loader8  = DataLoader(test_dataset8,  batch_size=Config.BATCH_SIZE, shuffle=False,
                                num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)

    model_multi = LogisticRegressionModel(input_dim, num_classes=8).to(Config.DEVICE)

    if Config.USE_CLASS_WEIGHTS:
        class_weights8 = compute_class_weights(y8_train, num_classes=8).to(Config.DEVICE)
        logger.info(f"Class weights: {class_weights8.cpu().numpy()}")
    else:
        class_weights8 = None

    criterion8 = nn.CrossEntropyLoss(weight=class_weights8)
    optimizer8 = optim.AdamW(model_multi.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

    try:
        scheduler8 = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer8, mode='min', factor=Config.LR_REDUCE_FACTOR,
            patience=Config.LR_REDUCE_PATIENCE, verbose=True
        )
    except TypeError:
        scheduler8 = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer8, mode='min', factor=Config.LR_REDUCE_FACTOR,
            patience=Config.LR_REDUCE_PATIENCE
        )

    model_multi, history_multi = train_model(model_multi, train_loader8, val_loader8,
                                              criterion8, optimizer8, scheduler8,
                                              Config.DEVICE, Config.EPOCHS, Config.EARLY_STOP_PATIENCE,
                                              "8-Class_LogReg_2_8")

    multi_results = {}
    for name, loader in [('train', train_loader8), ('val', val_loader8), ('test', test_loader8)]:
        multi_results[name] = evaluate_model(model_multi, loader, Config.DEVICE, class_names_8,
                                              "8-Class Logistic Regression", name)

    if Config.SAVE_PLOTS:
        plot_feature_importance(model_multi, feature_names, class_names_8, "8-Class_LogReg_2_8")

    save_model(model_multi, "logreg_multiclass_2_8", {
        'type':         'multiclass_logistic',
        'input_dim':    input_dim,
        'num_classes':  8,
        'class_names':  class_names_8,
        'feature_names': feature_names,
        'results': {k: {kk: vv for kk, vv in v.items() if kk not in ['predictions', 'labels']}
                    for k, v in multi_results.items()},
        'best_val_f1':  history_multi['val_f1'][-1] if history_multi['val_f1'] else None,
        'class_weights': class_weights8.cpu().tolist() if class_weights8 is not None else None
    })

    results['multiclass'] = multi_results

    logger.info("\n" + "=" * 80)
    logger.info("LOGISTIC REGRESSION TRAINING COMPLETED  [2/8-CLASS PIPELINE]")
    logger.info("=" * 80)
    logger.info(f"Binary  - Test Acc: {results['binary']['test']['accuracy']:.4f}  |  Macro F1: {results['binary']['test']['macro_f1']:.4f}")
    logger.info(f"8-Class - Test Acc: {results['multiclass']['test']['accuracy']:.4f}  |  Macro F1: {results['multiclass']['test']['macro_f1']:.4f}")
    logger.info(f"8-Class - Weighted F1: {results['multiclass']['test']['weighted_f1']:.4f}")
    logger.info(f"\nModels saved in:  {Config.MODELS_DIR}/")
    logger.info(f"Results saved in: {Config.RESULTS_DIR}/")

    print("\nLogistic regression training complete.\n")

if __name__ == "__main__":
    try:
        run_logistic_regression()
    except Exception as e:
        logger.exception("Unhandled exception: %s", e)
        sys.exit(1)
