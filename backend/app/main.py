"""FastAPI backend for codec-based deepfake detection."""
from __future__ import annotations

import pickle
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Repo root (parent of backend/)
REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = REPO_ROOT / "src" / "features"
sys.path.insert(0, str(FEATURES_DIR))

from dac_pipeline import load_dac_model  # noqa: E402
from extract_hypothesis_features import extract_analysis  # noqa: E402

MODELS_DIR = REPO_ROOT / "models"
RF_PATH = MODELS_DIR / "deepfake_detector_rf.pkl"
FEATURE_COLS_PATH = MODELS_DIR / "feature_cols.pkl"

ALLOWED_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".webm"}


class WindowScore(BaseModel):
    start_ms: float
    end_ms: float
    recon_score: float
    token_score: float


class AnalyzeResponse(BaseModel):
    authenticity_score: float
    spoof_probability: float
    verdict: str
    confidence: float
    windows: list[WindowScore]


class HealthResponse(BaseModel):
    status: str


class AppState:
    dac_model = None
    classifier = None
    label_encoder = None
    feature_cols: list[str] = []
    spoof_index: int = 1


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not RF_PATH.exists():
        raise RuntimeError(f"Missing model: {RF_PATH}")
    if not FEATURE_COLS_PATH.exists():
        raise RuntimeError(f"Missing feature columns: {FEATURE_COLS_PATH}")

    with open(RF_PATH, "rb") as f:
        bundle = pickle.load(f)
    state.classifier = bundle["classifier"]
    state.label_encoder = bundle["label_encoder"]

    with open(FEATURE_COLS_PATH, "rb") as f:
        state.feature_cols = pickle.load(f)

    classes = list(state.label_encoder.classes_)
    if "spoof" not in classes:
        raise RuntimeError("label_encoder must include 'spoof' class")
    state.spoof_index = int(classes.index("spoof"))

    print(
        f"Ready: RF ({RF_PATH.name}), {len(state.feature_cols)} features, "
        f"classes={classes}"
    )
    yield
    state.dac_model = None
    state.classifier = None


app = FastAPI(
    title="Codec Deepfake Detector",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # local React
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    if state.classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return HealthResponse(status="ok")


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(audio: UploadFile = File(...)):
    if state.classifier is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    if state.dac_model is None:
        print("Loading DAC model...")
        state.dac_model = load_dac_model()

    suffix = Path(audio.filename or "upload.wav").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    try:
        contents = await audio.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty audio file")

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            features, windows = extract_analysis(state.dac_model, tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        missing = [c for c in state.feature_cols if c not in features]
        if missing:
            raise HTTPException(
                status_code=500,
                detail=f"Feature extraction missing {len(missing)} columns",
            )

        X = np.array(
            [[features[c] for c in state.feature_cols]],
            dtype=np.float32,
        )
        proba = state.classifier.predict_proba(X)[0]
        classes = list(state.label_encoder.classes_)
        bonafide_idx = classes.index("bonafide")

        spoof_probability = float(proba[state.spoof_index])
        authenticity_score = float(proba[bonafide_idx])
        confidence = float(max(proba))
        verdict = "FAKE" if spoof_probability >= 0.5 else "REAL"

        return AnalyzeResponse(
            authenticity_score=round(authenticity_score, 6),
            spoof_probability=round(spoof_probability, 6),
            verdict=verdict,
            confidence=round(confidence, 6),
            windows=[WindowScore(**w) for w in windows],
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
