"""
FastAPI Deployment -- 2/8-Class IoT Intrusion Detection
Binary (Attack/Benign) + 8-class attack family classification.

Run via main_api.py:
    uvicorn main_api:app --reload
"""

import io
import json
import asyncio
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ── Paths (file-relative so they work regardless of launch directory) ────────
BASE_DIR       = Path(__file__).resolve().parent.parent
STATIC_DIR     = Path(__file__).resolve().parent / "static"
MODELS_TRAINED = BASE_DIR / "models" / "trained"
MODELS_PREP    = BASE_DIR / "models" / "preprocessing"
METADATA_PATH  = BASE_DIR / "data" / "metadata_2_8.json"

# ── Load metadata ────────────────────────────────────────────────────────────
with open(METADATA_PATH) as f:
    METADATA = json.load(f)

FEATURE_NAMES = METADATA["feature_columns"]
NUM_FEATURES  = METADATA["num_features"]
CLASS_NAMES_2 = METADATA["class_names_2"]
CLASS_NAMES_8 = METADATA["class_names_8"]

# ── Load scaler ──────────────────────────────────────────────────────────────
SCALER = joblib.load(MODELS_PREP / "scaler_2_8.pkl")
FALLBACK_RAW_VALUES = pd.Series(
    np.asarray(getattr(SCALER, "mean_", np.zeros(NUM_FEATURES)), dtype=np.float32),
    index=FEATURE_NAMES,
)

MODELS = {}
_binary_rf   = MODELS_TRAINED / "binary_rf_2_8.pkl"
_multi_rf    = MODELS_TRAINED / "multiclass_rf_2_8.pkl"
if _binary_rf.exists() and _multi_rf.exists():
    MODELS["rf"] = {
        "binary":     joblib.load(_binary_rf),
        "multiclass": joblib.load(_multi_rf),
    }
MODELS["logistic"] = {
    "binary":     joblib.load(MODELS_TRAINED / "logreg_binary_2_8.pkl"),
    "multiclass": joblib.load(MODELS_TRAINED / "logreg_multiclass_2_8.pkl"),
}
DEFAULT_MODEL      = "rf" if "rf" in MODELS else "logistic"
PREDICT_CHUNK_SIZE = 5_000

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="2/8-Class IoT Intrusion Detection API",
    description=(
        "Binary detection (Attack/Benign) + 8-class family classification.\n\n"
        "**Models:** `rf` (Random Forest) | `logistic` (Logistic Regression)\n\n"
        "**Features:** 28 network flow features\n\n"
        "**Output per row:** binary label + attack family + confidence scores"
    ),
    version="1.0.0",
)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ── Internal helpers ─────────────────────────────────────────────────────────
def get_model_pair(model_key: str):
    key = model_key.lower()
    if key not in MODELS:
        if key == "rf":
            raise HTTPException(
                status_code=503,
                detail="Random Forest model not available in Docker version"
            )
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_key}'. Use 'rf' or 'logistic'.",
        )
    return MODELS[key]["binary"], MODELS[key]["multiclass"], key


def sanitize_feature_frame(df: pd.DataFrame):
    df = df.copy()
    df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]
    drop = [c for c in df.columns if c.lower() in ("label", "label_2", "label_8")]
    if drop:
        df = df.drop(columns=drop, errors="ignore")
    present = [c for c in df.columns if c in FEATURE_NAMES]
    if not present:
        raise HTTPException(status_code=422, detail="No expected feature columns found in input.")
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    X = df.reindex(columns=FEATURE_NAMES, copy=True)
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    invalid = X.isna()
    summary = {
        "rows_adjusted":           int(invalid.any(axis=1).sum()),
        "invalid_values_replaced": int(invalid.sum().sum()),
        "missing_features_filled": missing,
    }
    X = X.fillna(FALLBACK_RAW_VALUES).astype(np.float32)
    return X, summary


def _safe_idx(classes_list, pred):
    """Return the index of pred in classes_list, with fallback to argmax."""
    try:
        return classes_list.index(pred)
    except ValueError:
        return 0


def batch_predict(df: pd.DataFrame, binary_model, multi_model):
    X, summary = sanitize_feature_frame(df)
    b_classes = list(binary_model.classes_)
    m_classes = list(multi_model.classes_)
    results   = []

    for start in range(0, len(X), PREDICT_CHUNK_SIZE):
        stop     = start + PREDICT_CHUNK_SIZE
        X_scaled = SCALER.transform(X.iloc[start:stop])

        b_preds = binary_model.predict(X_scaled)
        b_proba = binary_model.predict_proba(X_scaled)
        m_preds = multi_model.predict(X_scaled)
        m_proba = multi_model.predict_proba(X_scaled)

        for offset, (bp, bpr, mp, mpr) in enumerate(zip(b_preds, b_proba, m_preds, m_proba)):
            bi = _safe_idx(b_classes, bp)
            mi = _safe_idx(m_classes, mp)
            results.append({
                "index":             start + offset,
                "binary_prediction": str(bp),
                "attack_family":     str(mp),
                "confidence_binary": round(float(bpr[bi]), 4),
                "confidence_attack": round(float(mpr[mi]), 4),
            })

    return results, summary


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health", summary="Health check — model info and feature list")
def health(model: str = Query(default=DEFAULT_MODEL, description="'rf' or 'logistic'")):
    get_model_pair(model)
    return {
        "status":        "ok",
        "model_loaded":  model,
        "pipeline":      "2/8-class",
        "num_features":  NUM_FEATURES,
        "class_names_2": CLASS_NAMES_2,
        "class_names_8": CLASS_NAMES_8,
        "feature_names": FEATURE_NAMES,
    }


@app.post(
    "/predict-csv",
    summary="Upload a CSV and get binary + 8-class predictions",
)
async def predict_csv(
    file:  UploadFile = File(..., description="CSV with 28 feature columns"),
    model: str        = Query(default=DEFAULT_MODEL, description="'rf' or 'logistic'"),
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content), low_memory=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=422, detail="CSV file is empty.")

    bm, mm, key = get_model_pair(model)
    try:
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as pool:
            results, summary = await loop.run_in_executor(pool, batch_predict, df, bm, mm)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return {
        "filename":          file.filename,
        "model_used":        key,
        "total_rows":        len(df),
        "input_adjustments": summary,
        "predictions":       results,
    }


# ── Startup message ──────────────────────────────────────────────────────────
@app.on_event("startup")
def startup_message():
    print("=" * 55)
    print("  2/8-Class IoT IDS API -- READY")
    print(f"  Default model : {DEFAULT_MODEL}")
    print(f"  Features      : {NUM_FEATURES}")
    print(f"  Binary        : {CLASS_NAMES_2}")
    print(f"  8-class       : {CLASS_NAMES_8}")
    print("=" * 55)
