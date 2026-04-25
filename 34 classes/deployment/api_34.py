"""
FastAPI Deployment -- 34-Class IoT Intrusion Detection
Run from project root:  uvicorn deployment.api_34:app --reload
Docs:                   http://127.0.0.1:8000/docs
"""

import io
import json
import asyncio
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ==============================================================
# PATHS -- resolved relative to this file so the API works
# regardless of which directory you launch uvicorn from.
# ==============================================================
BASE_DIR       = Path(__file__).resolve().parent.parent
MODELS_TRAINED = BASE_DIR / "models" / "trained"
MODELS_PREP    = BASE_DIR / "models" / "preprocessing"
METADATA_PATH  = BASE_DIR / "data" / "metadata_34.json"
STATIC_DIR     = Path(__file__).resolve().parent / "static"

# ==============================================================
# LOAD SHARED ARTIFACTS (once at startup, shared across requests)
# ==============================================================
with open(METADATA_PATH) as f:
    METADATA = json.load(f)

FEATURE_NAMES = METADATA["feature_names"]   # ordered list of 39 features
NUM_FEATURES  = METADATA["num_features"]
NUM_CLASSES   = METADATA["num_classes"]
CLASS_NAMES   = METADATA["class_names"]

SCALER  = joblib.load(MODELS_PREP / "scaler_34.pkl")
ENCODER = joblib.load(MODELS_PREP / "label_encoder_34.pkl")

MODELS = {}
_rf_path = MODELS_TRAINED / "rf_34.pkl"
if _rf_path.exists():
    MODELS["rf"] = joblib.load(_rf_path)
MODELS["logistic"] = joblib.load(MODELS_TRAINED / "logistic_34.pkl")
DEFAULT_MODEL = "rf" if "rf" in MODELS else "logistic"
PREDICT_CHUNK_SIZE = 5_000
FALLBACK_RAW_VALUES = pd.Series(
    np.asarray(getattr(SCALER, "mean_", np.zeros(NUM_FEATURES)), dtype=np.float32),
    index=FEATURE_NAMES,
)

# ==============================================================
# FASTAPI APP
# ==============================================================
app = FastAPI(
    title="34-Class IoT Intrusion Detection API",
    description=(
        "Predicts network attack type from traffic flow features.\n\n"
        "**Models available:** `rf` (Random Forest, default) | `logistic` (Logistic Regression)\n\n"
        "**Input:** 39 network flow features (see `/health` for full list)\n\n"
        "**Output:** predicted attack class + confidence + top-3 probabilities"
    ),
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ==============================================================
# INTERNAL HELPERS
# ==============================================================
def get_model(model_key: str):
    key = model_key.lower()
    if key not in MODELS:
        if key == "rf":
            raise HTTPException(
                status_code=503,
                detail="Random Forest model not available in Docker version"
            )
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model_key}'. Use 'rf' or 'logistic'."
        )
    return MODELS[key], key


def sanitize_feature_frame(df: pd.DataFrame):
    """
    Normalize headers, coerce all feature columns to numeric, replace
    NaN/inf values with safe defaults, and preserve training column order.
    """
    df = df.copy()
    df.columns = [str(col).replace("\ufeff", "").strip() for col in df.columns]

    label_cols = [col for col in df.columns if col.lower() == "label"]
    if label_cols:
        df = df.drop(columns=label_cols, errors="ignore")

    present_features = [col for col in df.columns if col in FEATURE_NAMES]
    if not present_features:
        raise HTTPException(
            status_code=422,
            detail="No expected feature columns were found in the input."
        )

    missing_features = [feature for feature in FEATURE_NAMES if feature not in df.columns]
    X = df.reindex(columns=FEATURE_NAMES, copy=True)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    invalid_mask = X.isna()
    summary = {
        "rows_adjusted": int(invalid_mask.any(axis=1).sum()),
        "invalid_values_replaced": int(invalid_mask.sum().sum()),
        "missing_features_filled": missing_features,
    }

    X = X.fillna(FALLBACK_RAW_VALUES).astype(np.float32)
    return X, summary


def validate_and_prepare(record: dict):
    """
    Check all 39 features are present, reorder to training order,
    apply StandardScaler. Returns a (1, 39) float32 array.
    Extra columns in the input are silently ignored.
    """
    X_df, summary = sanitize_feature_frame(pd.DataFrame([record]))
    X_scaled = SCALER.transform(X_df)
    return X_scaled, summary


def make_prediction(X_scaled: np.ndarray, model) -> dict:
    """
    Run predict + predict_proba, return class name, confidence, top-3.
    """
    pred_idx   = int(model.predict(X_scaled)[0])
    pred_class = ENCODER.inverse_transform([pred_idx])[0]

    proba      = model.predict_proba(X_scaled)[0]
    confidence = float(proba[pred_idx])

    top3_idx = np.argsort(proba)[::-1][:3]
    top3     = [
        {
            "class":       ENCODER.inverse_transform([int(i)])[0],
            "probability": round(float(proba[i]), 4),
        }
        for i in top3_idx
    ]

    return {
        "predicted_class":   pred_class,
        "confidence":        round(confidence, 4),
        "top_3_predictions": top3,
    }


def batch_from_dataframe(df: pd.DataFrame, model):
    """
    Vectorized batch prediction. Inputs are sanitized first so malformed
    CSV values do not crash the API.
    """
    X, summary = sanitize_feature_frame(df)
    results = []
    for start in range(0, len(X), PREDICT_CHUNK_SIZE):
        stop = start + PREDICT_CHUNK_SIZE
        X_chunk = X.iloc[start:stop]
        X_scaled = SCALER.transform(X_chunk)

        pred_indices = model.predict(X_scaled).astype(int)
        proba_matrix = model.predict_proba(X_scaled)

        for offset, (idx, proba) in enumerate(zip(pred_indices, proba_matrix)):
            conf = float(proba[idx])
            top3_idx = np.argsort(proba)[::-1][:3]
            top3 = [
                {
                    "class":       ENCODER.inverse_transform([int(j)])[0],
                    "probability": round(float(proba[j]), 4),
                }
                for j in top3_idx
            ]

            results.append({
                "index":             start + offset,
                "predicted_class":   ENCODER.inverse_transform([idx])[0],
                "confidence":        round(conf, 4),
                "top_3_predictions": top3,
            })
    return results, summary


# ==============================================================
# ENDPOINT 0 -- GET /  (serve dark UI)
# ==============================================================
@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse(str(STATIC_DIR / "index.html"))


# ==============================================================
# ENDPOINT 1 -- GET /health
# ==============================================================
@app.get("/health", summary="Health check — model info and feature list")
def health(
    model: str = Query(default=DEFAULT_MODEL, description="'rf' or 'logistic'")
):
    _, key = get_model(model)
    m = MODELS[key]
    return {
        "status":        "ok",
        "model_loaded":  key,
        "model_type":    type(m).__name__,
        "num_features":  NUM_FEATURES,
        "num_classes":   NUM_CLASSES,
        "feature_names": FEATURE_NAMES,
        "class_names":   CLASS_NAMES,
    }


# ==============================================================
# ENDPOINT 2 -- POST /predict  (single record as JSON)
# ==============================================================
@app.post("/predict", summary="Predict a single network flow (JSON)")
def predict(
    flow: dict,
    model: str = Query(default=DEFAULT_MODEL, description="'rf' or 'logistic'"),
):
    try:
        m, _ = get_model(model)
        X, summary = validate_and_prepare(flow)
        result = make_prediction(X, m)
        if summary["invalid_values_replaced"] or summary["missing_features_filled"]:
            result["input_adjustments"] = summary
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


# ==============================================================
# ENDPOINT 3 -- POST /predict-batch  (list of JSON records)
# ==============================================================
@app.post("/predict-batch", summary="Predict a batch of network flows (JSON list)")
def predict_batch(
    flows: List[dict],
    model: str = Query(default=DEFAULT_MODEL, description="'rf' or 'logistic'"),
):
    if not flows:
        raise HTTPException(status_code=422, detail="Empty list provided.")

    try:
        m, _ = get_model(model)
        results, _ = batch_from_dataframe(pd.DataFrame(flows), m)
        return results
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}") from exc


# ==============================================================
# ENDPOINT 4 -- POST /predict-csv
# Upload a CSV file -> auto-converted to records -> predictions
# ==============================================================
@app.post(
    "/predict-csv",
    summary="Upload a CSV file of network flows and get predictions",
    description=(
        "Upload a `.csv` file whose columns match the 39 required feature names.\n\n"
        "The API reads the file, validates columns, applies the scaler, "
        "and returns one prediction per row.\n\n"
        "A `Label` column is allowed in the CSV (e.g. from the original dataset) "
        "but is ignored during prediction."
    ),
)
async def predict_csv(
    file: UploadFile = File(..., description="CSV file with 39 feature columns"),
    model: str = Query(default=DEFAULT_MODEL, description="'rf' or 'logistic'"),
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content), low_memory=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=422, detail="CSV file is empty.")

    try:
        m, key = get_model(model)
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as pool:
            results, summary = await loop.run_in_executor(pool, batch_from_dataframe, df, m)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"CSV prediction failed: {exc}") from exc

    return {
        "filename":          file.filename,
        "model_used":        key,
        "total_rows":        len(df),
        "input_adjustments": summary,
        "predictions":       results,
    }


# ==============================================================
# STARTUP MESSAGE
# ==============================================================
@app.on_event("startup")
def startup_message():
    print("=" * 55)
    print("  34-Class IoT Intrusion Detection API -- READY")
    print(f"  Default model : {DEFAULT_MODEL}")
    print(f"  Features      : {NUM_FEATURES}")
    print(f"  Classes       : {NUM_CLASSES}")
    print("  Docs          : http://127.0.0.1:8000/docs")
    print("=" * 55)
