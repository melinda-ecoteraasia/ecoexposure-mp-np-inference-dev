import os
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client

from preprocessing import FEATURE_ORDER, preprocess_and_measure
from model_router import load_bundle, available_models

APP_VERSION = "ecoexposure-mp-np-inference-v1"

SUPABASE_URL = os.getenv("SUPABASE_URL","").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY","").strip()
ALLOWED_ORIGINS = [
    x.strip() for x in os.getenv(
        "ALLOWED_ORIGINS",
        "https://melinda-ecoteraasia.github.io,http://localhost:8000,http://127.0.0.1:8000"
    ).split(",") if x.strip()
]

app = FastAPI(title="EcoExposure MP + NP Inference API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    sample_id: str
    water_type: str
    model_name: str
    top_photo_path: str
    storage_bucket: str = "ecoexposure_images"

def get_supabase():
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(status_code=503, detail="Supabase backend credentials are not configured on Render.")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

def run_one_model(model_name, metrics):
    bundle = load_bundle(model_name)
    feature_names = bundle.get("feature_names") or FEATURE_ORDER
    model = bundle["model"]
    X = pd.DataFrame([{name: metrics.get(name) for name in feature_names}], columns=feature_names)

    prediction = model.predict(X)[0]
    confidence = None
    class_probabilities = None

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        classes = [str(x) for x in model.classes_]
        confidence = float(np.max(probs))
        class_probabilities = {classes[i]: float(probs[i]) for i in range(len(probs))}

    return {
        "model_name": model_name,
        "model_version": bundle.get("model_version", model_name),
        "prediction": str(prediction),
        "confidence": confidence,
        "class_probabilities": class_probabilities,
        "display_labels": bundle.get("display_labels"),
    }

@app.get("/")
def root():
    return {
        "service": "EcoExposure MP + NP Inference API",
        "version": APP_VERSION,
        "status": "ok",
        "models": available_models(),
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "models": available_models(),
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY),
    }

@app.post("/predict")
def predict(request: PredictRequest, authorization: Optional[str] = Header(default=None)):
    sb = get_supabase()

    try:
        image_bytes = sb.storage.from_(request.storage_bucket).download(request.top_photo_path)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Storage download failed: {exc}")

    try:
        # IMPORTANT: preprocess ONCE, then send the same frozen 18 metrics to MP and NP.
        metrics, qc = preprocess_and_measure(image_bytes, filename=request.top_photo_path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Image preprocessing failed: {exc}")

    try:
        mp_result = run_one_model(request.model_name, metrics)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"MP prediction failed: {exc}")

    # NP model is matrix-independent in this development version.
    try:
        np_result = run_one_model("np_v1", metrics)
    except FileNotFoundError:
        np_result = None
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"NP prediction failed: {exc}")

    # Preserve the original MP fields for backward compatibility with the page.
    try:
        predicted_mp = int(float(mp_result["prediction"]))
    except Exception:
        predicted_mp = mp_result["prediction"]

    index = None
    band = None
    if isinstance(predicted_mp, int):
        if predicted_mp <= 25:
            index, band = 1, "Green"
        elif predicted_mp <= 50:
            index, band = 2, "Green-Yellow"
        elif predicted_mp <= 75:
            index, band = 3, "Yellow"
        elif predicted_mp <= 100:
            index, band = 4, "Orange"
        else:
            index, band = 5, "Red"

    return {
        "sample_id": request.sample_id,
        "water_type": request.water_type,

        "model_name": mp_result["model_name"],
        "model_version": mp_result["model_version"],
        "predicted_mp": predicted_mp,
        "ecoexposure_index": index,
        "band": band,
        "confidence": mp_result["confidence"],
        "class_probabilities": mp_result["class_probabilities"],

        "mp": mp_result,
        "np": np_result,

        "metrics": metrics,
        "qc": qc,
        "api_version": APP_VERSION,
    }
