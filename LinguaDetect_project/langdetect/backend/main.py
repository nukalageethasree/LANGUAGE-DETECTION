"""
Language Detection API — FastAPI Backend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import os

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Language Detector API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Language metadata (17 classes from the Kaggle dataset) ────────────────────
LANGUAGE_META = {
    "Arabic":     {"flag": "🇸🇦", "native": "العربية",      "color": "#10B981"},
    "Danish":     {"flag": "🇩🇰", "native": "Dansk",         "color": "#3B82F6"},
    "Dutch":      {"flag": "🇳🇱", "native": "Nederlands",    "color": "#F97316"},
    "English":    {"flag": "🇬🇧", "native": "English",       "color": "#6366F1"},
    "French":     {"flag": "🇫🇷", "native": "Français",      "color": "#EC4899"},
    "German":     {"flag": "🇩🇪", "native": "Deutsch",       "color": "#F59E0B"},
    "Greek":      {"flag": "🇬🇷", "native": "Ελληνικά",      "color": "#0EA5E9"},
    "Hindi":      {"flag": "🇮🇳", "native": "हिन्दी",         "color": "#EF4444"},
    "Italian":    {"flag": "🇮🇹", "native": "Italiano",      "color": "#14B8A6"},
    "Kannada":    {"flag": "🇮🇳", "native": "ಕನ್ನಡ",           "color": "#8B5CF6"},
    "Malayalam":  {"flag": "🇮🇳", "native": "മലയാളം",         "color": "#F43F5E"},
    "Portugese":  {"flag": "🇵🇹", "native": "Português",     "color": "#22C55E"},
    "Russian":    {"flag": "🇷🇺", "native": "Русский",       "color": "#A855F7"},
    "Spanish":    {"flag": "🇪🇸", "native": "Español",       "color": "#EAB308"},
    "Swedish":    {"flag": "🇸🇪", "native": "Svenska",       "color": "#06B6D4"},
    "Tamil":      {"flag": "🇮🇳", "native": "தமிழ்",          "color": "#F97316"},
    "Turkish":    {"flag": "🇹🇷", "native": "Türkçe",        "color": "#DC2626"},
}

# Ordered class list (must match training label encoder order — alphabetical)
CLASSES = sorted(LANGUAGE_META.keys())

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL = None
MAX_LEN = 300

def load_model():
    global MODEL
    model_path = os.path.join(os.path.dirname(__file__), "..", "model", "language_detector.keras")
    model_path = os.path.abspath(model_path)
    print(f"Loading model from: {model_path}")
    try:
        import tensorflow as tf
        MODEL = tf.keras.models.load_model(model_path)
        print(f"✅  Model loaded — input: {MODEL.input_shape}, output: {MODEL.output_shape}")
    except Exception as e:
        print(f"❌  Model load error: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    load_model()

# ── Preprocessing (mirrors training: UTF-8 byte sequences) ───────────────────
def text_to_sequence(text: str, maxlen: int = MAX_LEN) -> np.ndarray:
    encoded = [min(b, 255) for b in text.encode("utf-8", errors="replace")]
    if len(encoded) >= maxlen:
        seq = encoded[:maxlen]
    else:
        seq = encoded + [0] * (maxlen - len(encoded))
    return np.array(seq, dtype="int32")

# ── Request / Response schemas ────────────────────────────────────────────────
class DetectRequest(BaseModel):
    text: str

class LanguagePrediction(BaseModel):
    language: str
    confidence: float
    flag: str
    native_name: str
    color: str
    rank: int

class DetectResponse(BaseModel):
    detected: LanguagePrediction
    top_predictions: list[LanguagePrediction]
    char_count: int
    word_count: int

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    return FileResponse(os.path.abspath(frontend_path))

@app.post("/detect", response_model=DetectResponse)
async def detect_language(req: DetectRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    if len(text) > 5000:
        raise HTTPException(status_code=400, detail="Text too long (max 5000 chars).")

    seq = text_to_sequence(text)
    seq_batch = np.expand_dims(seq, axis=0)

    probs = MODEL.predict(seq_batch, verbose=0)[0]

    # Build ranked predictions
    top_indices = np.argsort(probs)[::-1][:5]
    predictions = []
    for rank, idx in enumerate(top_indices, start=1):
        lang = CLASSES[idx]
        meta = LANGUAGE_META.get(lang, {"flag": "🌐", "native": lang, "color": "#6B7280"})
        predictions.append(LanguagePrediction(
            language    = lang,
            confidence  = float(round(probs[idx] * 100, 2)),
            flag        = meta["flag"],
            native_name = meta["native"],
            color       = meta["color"],
            rank        = rank,
        ))

    return DetectResponse(
        detected        = predictions[0],
        top_predictions = predictions,
        char_count      = len(text),
        word_count      = len(text.split()),
    )

@app.get("/languages")
async def list_languages():
    return [
        {"name": lang, **LANGUAGE_META[lang]}
        for lang in CLASSES
    ]

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": MODEL is not None}

# ── Serve frontend static files ───────────────────────────────────────────────
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.isdir(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
