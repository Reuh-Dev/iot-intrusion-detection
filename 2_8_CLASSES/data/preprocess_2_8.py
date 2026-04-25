"""Preprocessing for CICIoT2023 (2/8-class pipeline). Run from 2_8_CLASSES/ directory."""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import glob

_SCRIPT_DIR = Path(__file__).resolve().parent        # 2_8_CLASSES/preprocessing/
_BASE_DIR   = _SCRIPT_DIR.parent                     # 2_8_CLASSES/
_PROJECT_ROOT = _BASE_DIR.parent                     # PROJECT-FINAL-V/ (where Merged*.csv live)

class Config:
    DATA_DIR = _BASE_DIR / "data" / "raw"

    OUTPUT_DIR   = _BASE_DIR / "data" / "processed_data_2_8"
    MODELS_DIR   = _BASE_DIR / "models" / "preprocessing"
    METADATA_DIR = _BASE_DIR / "data"

    FILE_PATTERN = "Merged0[1-6].csv"

    RANDOM_STATE = 42
    TEST_SIZE    = 0.2
    VAL_SIZE     = 0.5  # From remaining temp data

    LOG_LEVEL = logging.INFO

    DROP_DUPLICATES = False  # Based on inspection: 42.9% duplicates

    @classmethod
    def setup_directories(cls):
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cls.METADATA_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("Preprocessing")
    logger.setLevel(Config.LOG_LEVEL)

    console = logging.StreamHandler()
    console.setLevel(Config.LOG_LEVEL)

    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log_file = Config.OUTPUT_DIR / f"preprocess_{datetime.now():%Y%m%d_%H%M%S}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger

logger = setup_logging()

LABEL_MAPPING = {
    # DDoS
    'DDOS-PSHACK_FLOOD': 'DDoS', 'DDOS-ICMP_FLOOD': 'DDoS',
    'DDOS-TCP_FLOOD': 'DDoS', 'DDOS-SYN_FLOOD': 'DDoS',
    'DDOS-UDP_FLOOD': 'DDoS', 'DDOS-SYNONYMOUSIP_FLOOD': 'DDoS',
    'DDOS-RSTFINFLOOD': 'DDoS', 'DDOS-SLOWLORIS': 'DDoS',
    'DDOS-ICMP_FRAGMENTATION': 'DDoS', 'DDOS-ACK_FRAGMENTATION': 'DDoS',
    'DDOS-UDP_FRAGMENTATION': 'DDoS', 'DDOS-HTTP_FLOOD': 'DDoS',
    # DoS
    'DOS-UDP_FLOOD': 'DoS', 'DOS-TCP_FLOOD': 'DoS',
    'DOS-SYN_FLOOD': 'DoS', 'DOS-HTTP_FLOOD': 'DoS',
    # Mirai
    'MIRAI-GREIP_FLOOD': 'Mirai', 'MIRAI-GREETH_FLOOD': 'Mirai',
    'MIRAI-UDPPLAIN': 'Mirai',
    # Recon
    'RECON-HOSTDISCOVERY': 'Recon', 'RECON-PORTSCAN': 'Recon',
    'RECON-OSSCAN': 'Recon', 'RECON-PINGSWEEP': 'Recon',
    'VULNERABILITYSCAN': 'Recon',
    # Web
    'BACKDOOR_MALWARE': 'Web', 'XSS': 'Web', 'BROWSERHIJACKING': 'Web',
    'SQLINJECTION': 'Web', 'COMMANDINJECTION': 'Web', 'UPLOADING_ATTACK': 'Web',
    # BruteForce
    'DICTIONARYBRUTEFORCE': 'BruteForce',
    # Spoofing
    'DNS_SPOOFING': 'Spoofing', 'MITM-ARPSPOOFING': 'Spoofing',
    # Benign
    'BENIGN': 'Benign'
}

# High correlation clusters — dropped based on inspection
CORRELATED_FEATURES_TO_DROP = [
    'fin_flag_number', 'rst_flag_number', 'fin_count',
    'psh_flag_number', 'syn_flag_number', 'IPv', 'LLC', 'Tot size'
]

# Features requiring log transformation (skewness > 1, from inspection)
LOG_TRANSFORM_FEATURES = [
    'Protocol Type', 'Time_To_Live', 'Rate', 'fin_flag_number',
    'syn_flag_number', 'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
    'ece_flag_number', 'cwr_flag_number', 'ack_count', 'syn_count',
    'fin_count', 'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP',
    'SSH', 'IRC', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv', 'LLC',
    'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Variance'
]

ADDITIONAL_DROPS = ['ARP', 'Telnet', 'SMTP']  # Low importance / domain knowledge


def load_data() -> pd.DataFrame:
    logger.info("Loading data files...")

    files = sorted(glob.glob(str(Config.DATA_DIR / Config.FILE_PATTERN)))
    logger.info(f"Found {len(files)} files: {[Path(f).name for f in files]}")

    df_list = []
    total_rows = 0

    for file in files:
        chunk = pd.read_csv(file)
        chunk = chunk.dropna(subset=['Label'])
        df_list.append(chunk)
        total_rows += len(chunk)
        logger.debug(f"Loaded {Path(file).name}: {len(chunk):,} rows")

    df = pd.concat(df_list, ignore_index=True)
    logger.info(f"Combined shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    initial_shape = df.shape
    logger.info(f"Initial shape: {initial_shape[0]:,} x {initial_shape[1]}")

    # 1. Handle infinite values
    logger.info("Handling infinite values...")
    inf_cols = df.select_dtypes(include=[np.number]).columns[
        df.select_dtypes(include=[np.number]).apply(lambda x: np.isinf(x).any())
    ].tolist()

    if inf_cols:
        logger.warning(f"Found infinite values in columns: {inf_cols}")
        df = df.replace([np.inf, -np.inf], np.nan)

    # 2. Handle missing values
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        logger.info(f"Missing values: {missing_before}")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)
                logger.debug(f"Imputed {col} with median: {median_val:.2f}")

        remaining_missing = int(df.isnull().sum().sum())
        if remaining_missing > 0:
            before_drop = len(df)
            df = df.dropna()
            logger.info(f"Dropped {before_drop - len(df):,} additional NaN rows")

    # 3. Remove duplicates
    if Config.DROP_DUPLICATES:
        before_dup = len(df)
        df = df.drop_duplicates()
        dup_removed = before_dup - len(df)
        logger.info(f"Removed {dup_removed:,} duplicate rows ({dup_removed/before_dup*100:.1f}%)")

    logger.info(f"Final shape after cleaning: {df.shape[0]:,} x {df.shape[1]}")

    return df


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Creating label encodings...")

    df['Label_8'] = df['Label'].map(LABEL_MAPPING)
    df['Label_2'] = df['Label'].apply(lambda x: 'Benign' if x == 'BENIGN' else 'Attack')

    unmapped = df[df['Label_8'].isna()]['Label'].unique()
    if len(unmapped) > 0:
        logger.error(f"Unmapped labels found: {unmapped}")
        raise ValueError(f"Missing mappings for labels: {unmapped}")

    class_dist = df['Label_8'].value_counts()
    logger.info("8-Class distribution:")
    for cls, count in class_dist.items():
        logger.info(f"  {cls}: {count:,} ({count/len(df)*100:.2f}%)")

    return df

def apply_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Applying log transformations...")

    transform_cols = [col for col in LOG_TRANSFORM_FEATURES if col in df.columns]
    logger.info(f"Transforming {len(transform_cols)} columns")

    for col in transform_cols:
        min_val = df[col].min()
        if min_val < 0:
            logger.warning(f"Column {col} has negative values (min={min_val}). Shifting by {-min_val + 1}")
            df[col] = np.log1p(df[col] - min_val + 1)
        else:
            df[col] = np.log1p(df[col])

        logger.debug(f"Applied log1p to {col}")

    return df

def drop_redundant_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Dropping redundant features...")

    drops = CORRELATED_FEATURES_TO_DROP + ADDITIONAL_DROPS
    drops = [col for col in drops if col in df.columns]

    logger.info(f"Dropping {len(drops)} columns: {drops}")
    df = df.drop(columns=drops)

    logger.info(f"Remaining features: {len(df.columns) - 3}")  # -3 for Label columns

    return df


def create_splits(df: pd.DataFrame) -> Tuple:
    logger.info("Creating stratified splits...")

    feature_cols = [col for col in df.columns if col not in ['Label', 'Label_8', 'Label_2']]
    X  = df[feature_cols]
    y8 = df['Label_8']
    y2 = df['Label_2']

    logger.info(f"Features: {len(feature_cols)} columns")

    # First split: train (80%) and temp (20%)
    X_train, X_temp, y8_train, y8_temp, y2_train, y2_temp = train_test_split(
        X, y8, y2,
        test_size=Config.TEST_SIZE,
        stratify=y8,
        random_state=Config.RANDOM_STATE
    )

    # Second split: val (10%) and test (10%)
    X_val, X_test, y8_val, y8_test, y2_val, y2_test = train_test_split(
        X_temp, y8_temp, y2_temp,
        test_size=Config.VAL_SIZE,
        stratify=y8_temp,
        random_state=Config.RANDOM_STATE
    )

    logger.info(f"Train set: {len(X_train):,} rows")
    logger.info(f"Val set:   {len(X_val):,} rows")
    logger.info(f"Test set:  {len(X_test):,} rows")

    for split_name, y_split in [('Train', y8_train), ('Val', y8_val), ('Test', y8_test)]:
        dist = y_split.value_counts(normalize=True)
        logger.debug(f"{split_name} class distribution: {dist.to_dict()}")

    return (X_train, X_val, X_test,
            y8_train, y8_val, y8_test,
            y2_train, y2_val, y2_test,
            feature_cols)


def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> Tuple:
    logger.info("Scaling features...")

    scaler = StandardScaler()

    # Fit only on training data
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled   = pd.DataFrame(X_val_scaled,   columns=X_val.columns,   index=X_val.index)
    X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=X_test.columns,  index=X_test.index)

    scaler_path = Config.MODELS_DIR / "scaler_2_8.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")

    logger.debug(f"Feature means: {scaler.mean_[:5]}...")
    logger.debug(f"Feature scales: {scaler.scale_[:5]}...")

    return X_train_scaled, X_val_scaled, X_test_scaled


def save_label_encoders(y2: pd.Series, y8: pd.Series):
    logger.info("Saving label encoders...")

    label_encoder_2 = LabelEncoder()
    label_encoder_8 = LabelEncoder()
    label_encoder_2.fit(y2)
    label_encoder_8.fit(y8)

    encoder_bundle = {
        'label_encoder_2': label_encoder_2,
        'label_encoder_8': label_encoder_8,
    }

    encoder_path = Config.MODELS_DIR / "label_encoder_2_8.pkl"
    joblib.dump(encoder_bundle, encoder_path)
    logger.info(f"Label encoders saved to {encoder_path}")


def save_splits(X_train, X_val, X_test, y8_train, y8_val, y8_test,
                y2_train, y2_val, y2_test, feature_cols: List[str]):
    logger.info("Saving processed splits...")

    X_train.to_parquet(Config.OUTPUT_DIR / "X_train.parquet",  index=False, compression='snappy')
    X_val.to_parquet(Config.OUTPUT_DIR   / "X_val.parquet",    index=False, compression='snappy')
    X_test.to_parquet(Config.OUTPUT_DIR  / "X_test.parquet",   index=False, compression='snappy')

    y8_train.to_frame(name='Label_8').to_parquet(Config.OUTPUT_DIR / "y8_train.parquet", index=False)
    y8_val.to_frame(name='Label_8').to_parquet(Config.OUTPUT_DIR   / "y8_val.parquet",   index=False)
    y8_test.to_frame(name='Label_8').to_parquet(Config.OUTPUT_DIR  / "y8_test.parquet",  index=False)

    y2_train.to_frame(name='Label_2').to_parquet(Config.OUTPUT_DIR / "y2_train.parquet", index=False)
    y2_val.to_frame(name='Label_2').to_parquet(Config.OUTPUT_DIR   / "y2_val.parquet",   index=False)
    y2_test.to_frame(name='Label_2').to_parquet(Config.OUTPUT_DIR  / "y2_test.parquet",  index=False)

    metadata = {
        'feature_columns': feature_cols,
        'num_features': len(feature_cols),
        'num_classes_8': 8,
        'num_classes_2': 2,
        'class_names_8': sorted(y8_train.unique()),
        'class_names_2': sorted(y2_train.unique()),
        'scaler_path': str(Config.MODELS_DIR / "scaler_2_8.pkl"),
        'encoder_path': str(Config.MODELS_DIR / "label_encoder_2_8.pkl"),
        'source_data': 'Merged01-06.csv (CICIoT2023)',
        'creation_date': datetime.now().isoformat(),
        'train_size': len(X_train),
        'val_size':   len(X_val),
        'test_size':  len(X_test)
    }

    meta_dest = Config.METADATA_DIR / "metadata_2_8.json"
    with open(meta_dest, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"All files saved to {Config.OUTPUT_DIR}/")
    logger.info(f"Metadata saved to {meta_dest}")
    logger.info(f"Metadata saved with {len(feature_cols)} features")


def validate_output(X_train, X_val, X_test, y8_train, y8_val, y8_test):
    logger.info("Validating output...")

    checks_passed = True

    for name, data in [('Train', X_train), ('Val', X_val), ('Test', X_test)]:
        if data.isnull().any().any():
            logger.error(f"NaN values found in {name} set")
            checks_passed = False
        if np.isinf(data.values).any():
            logger.error(f"Inf values found in {name} set")
            checks_passed = False

    train_dist = y8_train.value_counts(normalize=True)
    val_dist   = y8_val.value_counts(normalize=True)

    max_diff = (train_dist - val_dist).abs().max()
    if max_diff > 0.02:
        logger.warning(f"Class distribution mismatch between train and val: max diff {max_diff:.3f}")

    zero_var_cols = X_train.columns[X_train.var() == 0].tolist()
    if zero_var_cols:
        logger.warning(f"Zero variance columns found: {zero_var_cols}")

    if checks_passed:
        logger.info("All validation checks passed")
    else:
        logger.error("Validation failed - check logs for details")

    return checks_passed


def main():
    logger.info("=" * 80)
    logger.info("PRODUCTION DATA PREPROCESSING FOR CICIoT2023  [2/8-CLASS PIPELINE]")
    logger.info("=" * 80)

    Config.setup_directories()

    try:
        df = load_data()
        df = clean_data(df)
        df = create_labels(df)
        df = apply_log_transforms(df)
        df = drop_redundant_features(df)

        (X_train, X_val, X_test,
         y8_train, y8_val, y8_test,
         y2_train, y2_val, y2_test,
         feature_cols) = create_splits(df)

        save_label_encoders(y2_train, y8_train)

        X_train_scaled, X_val_scaled, X_test_scaled = scale_features(X_train, X_val, X_test)

        validate_output(X_train_scaled, X_val_scaled, X_test_scaled,
                        y8_train, y8_val, y8_test)

        save_splits(X_train_scaled, X_val_scaled, X_test_scaled,
                    y8_train, y8_val, y8_test,
                    y2_train, y2_val, y2_test,
                    feature_cols)

        logger.info("\n" + "=" * 80)
        logger.info("PREPROCESSING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Total samples: {len(df):,}")
        logger.info(f"Features: {len(feature_cols)}")
        logger.info(f"Output directory: {Config.OUTPUT_DIR}/")

    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
