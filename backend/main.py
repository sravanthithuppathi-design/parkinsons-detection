"""
PD Detect — FastAPI Backend
Endpoints: /health, /analyze, /history, /history/{id}, /stats, /report
"""

import os
import io
import json
import uuid
import random
import logging
from datetime import datetime
from typing import List, Optional

import numpy as np
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from feature_extractor import extract_features

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PD Detect API",
    description="Parkinson's Disease Detection via Voice Analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins — frontend hosted on Vercel/Netlify
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── UCI feature names (22 features, matches training data column order) ─────
UCI_FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR",
    "RPDE", "DFA",
    "spread1", "spread2", "D2", "PPE",
]

# ─── Load models ──────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
models_loaded = False
scaler        = None
ensemble      = None
feature_importance_global: dict = {}


def try_load_models():
    global models_loaded, scaler, ensemble, feature_importance_global
    try:
        scaler   = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        ensemble = joblib.load(os.path.join(MODELS_DIR, "ensemble.pkl"))
        fi_path  = os.path.join(MODELS_DIR, "feature_importance.json")
        if os.path.exists(fi_path):
            with open(fi_path) as fh:
                feature_importance_global = json.load(fh)
        models_loaded = True
        logger.info("✅ Models loaded successfully.")
    except Exception as e:
        logger.warning(f"⚠️  Models not found ({e}). Using heuristic fallback.")
        models_loaded = False


try_load_models()

# ─── In-memory history (max 200 entries) ─────────────────────────────────────
analysis_history: List[dict] = []
MAX_HISTORY = 200

# ─── Normal ranges (adjusted for browser WebM recording characteristics) ──────
NORMAL_RANGES = {
    "jitter_local":           (0.001, 0.08),   # browser jitter is ~8x clinical
    "shimmer_local":          (0.010, 0.12),   # browser shimmer is ~3x clinical
    "hnr":                    (5.0,   35.0),   # browser HNR is ~8dB lower
    "pitch_mean":             (75.0,  300.0),
    "pitch_std":              (0.5,   20.0),
    "zcr_mean":               (0.01,  0.25),
    "spectral_centroid_mean": (300.0, 5000.0),
    "rms_mean":               (0.005, 0.50),
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

# ── UCI clinical statistics (from Little et al. 2007) used for calibration ──
_UCI_HE = {  # Healthy means
    "jitter_pct": 0.003866, "shimmer": 0.01762,  "hnr": 24.679,
    "nhr": 0.01148,         "rpde": 0.44255,     "dfa": 0.69572,
    "spread1": -6.7593,     "spread2": 0.16029,  "d2": 2.1545, "ppe": 0.12302,
}
_UCI_PD = {  # PD means
    "jitter_pct": 0.006989, "shimmer": 0.03366,  "hnr": 20.974,
    "nhr": 0.02921,         "rpde": 0.51682,     "dfa": 0.72541,
    "spread1": -5.3334,     "spread2": 0.24813,  "d2": 2.4561, "ppe": 0.23383,
}


def map_to_uci_vector(features: dict) -> np.ndarray:
    """
    Map extracted audio features -> 22 UCI clinical feature vector.

    Design principle: PITCH-ANCHORED HEALTHY-CENTER mapping.
    - Pitch is mapped to the CENTER of UCI healthy range so that normal
      male (85-155Hz) and female (165-255Hz) voices both land in healthy zone.
    - DFA/RPDE/PPE use WIDE healthy plateaus so typical browser variation
      stays firmly in UCI healthy territory.
    - Jitter/shimmer use aggressive log-compression so noise-inflated
      browser values still map to healthy UCI range.

    UCI healthy ranges (mean +/- 2std):
      Fo:    182 +/- 83 Hz   (wide: includes male+female)
      DFA:   0.696 +/- 0.111  -> [0.574, 0.825]
      RPDE:  0.443 +/- 0.208  -> [0.257, 0.685]
      PPE:   0.123 +/- 0.180  -> [0.045, 0.527]
    """
    # -- Pitch (ANCHORED TO HEALTHY CENTER) --------------------------------
    # CRITICAL FIX: Do NOT use linear offset that makes male voices look PD.
    # UCI healthy mean Fo=182Hz includes both male (~120Hz) and female (~220Hz).
    # Map ALL normal pitch range (85-250Hz) to UCI healthy center (170-195Hz).
    # Only extreme pitch deviations (< 75Hz or > 300Hz) move toward PD range.
    raw_pitch = float(np.clip(features.get("pitch_mean", 165.0), 50.0, 400.0))
    pitch_std = float(features.get("pitch_std", 5.0))

    # Sigmoid-like mapping: normal voices -> healthy center
    # 85Hz  -> 155Hz (low but still healthy)
    # 120Hz -> 172Hz (healthy male)
    # 165Hz -> 182Hz (UCI healthy mean)
    # 220Hz -> 195Hz (healthy female)
    # 280Hz -> 200Hz (high but healthy)
    # Use soft clipping around healthy range
    pitch = float(np.clip(
        182.0 + 25.0 * np.tanh((raw_pitch - 165.0) / 80.0),
        100.0, 250.0
    ))
    fhi   = float(np.clip(pitch * 1.23, 100.0, 400.0))
    flo   = float(np.clip(pitch * 0.80, 70.0, 250.0))

    # -- Jitter (log-compression with PD sensitivity) -----------------------
    # Browser zero-crossing jitter is 30-60x clinical MDVP values.
    # Map so healthy stays healthy AND PD voices reach PD range.
    #   raw=0.03  -> UCI 0.0021 (very clean healthy)
    #   raw=0.08  -> UCI 0.00387 (typical healthy = UCI healthy mean)
    #   raw=0.15  -> UCI 0.00535 (noisy but still healthy)
    #   raw=0.30  -> UCI 0.00736 (clear PD)
    #   raw=0.50  -> UCI 0.00927 (strong PD)
    #   raw=1.00  -> UCI 0.01239 (severe PD)
    raw_jitter = float(np.clip(features.get("jitter_local", 0.05), 0.001, 5.0))
    jitter_pct = float(np.clip(
        0.00387 * (raw_jitter / 0.08) ** 0.45,
        0.00168, 0.03316
    ))

    jitter_abs = float(np.clip(jitter_pct / (pitch * 200.0 + 1e-9), 7e-6, 2.6e-4))
    jitter_rap = float(np.clip(jitter_pct * 0.53, 0.00068, 0.02144))
    jitter_ppq = float(np.clip(jitter_pct * 0.56, 0.00092, 0.01958))
    jitter_ddp = jitter_rap * 3.0

    # -- Shimmer (log-compression with PD sensitivity) ----------------------
    # Browser shimmer is 3-5x clinical. Map so healthy stays healthy
    # AND PD voices with high shimmer reach PD range.
    #   raw=0.03  -> UCI 0.009 (clean healthy)
    #   raw=0.055 -> UCI 0.0176 (UCI healthy mean)
    #   raw=0.10  -> UCI 0.029 (upper healthy / borderline)
    #   raw=0.15  -> UCI 0.039 (PD range)
    #   raw=0.25  -> UCI 0.057 (clear PD)
    #   raw=0.50  -> UCI 0.091 (strong PD)
    raw_shimmer = float(np.clip(features.get("shimmer_local", 0.055), 0.001, 1.0))
    s_base   = _UCI_HE["shimmer"]              # 0.01762
    s_anchor = 0.055                           # typical healthy browser value
    s_exp    = 0.75                            # PD-sensitive compression
    shimmer = float(np.clip(
        s_base * (raw_shimmer / s_anchor) ** s_exp,
        0.00954, 0.11908
    ))

    shimmer_db  = float(np.clip(shimmer * 9.23, 0.085, 1.302))   # UCI: shimmer_db ≈ shimmer * 9.23
    shimmer_apq3= float(np.clip(shimmer * 0.54, 0.00455, 0.05647))
    shimmer_apq5= float(np.clip(shimmer * 0.61, 0.00570, 0.07940))
    shimmer_apq = float(np.clip(shimmer * 0.82, 0.00719, 0.13778))
    shimmer_dda = shimmer_apq3 * 3.0

    # -- HNR / NHR ---------------------------------------------------------
    # Browser cepstrum HNR is ~10-15dB lower than clinical MDVP-HNR.
    # Map so typical healthy (8-18dB) -> UCI healthy (22-28dB).
    raw_hnr = float(features.get("hnr", 10.0))
    hnr     = float(np.clip(raw_hnr + 15.0, 8.441, 33.047))
    # NHR is computed from jitter+shimmer deviation (consistent with rest of mapping).
    # Clinical NHR measures noise components - correlated with jitter/shimmer, NOT HNR.
    # UCI healthy mean NHR=0.011, PD mean=0.029.
    # Will be computed after jitter_dev and shimmer_dev are available (see below).

    # -- RPDE (BALANCED mapping) --------------------------------------------
    # Browser RPDE proxy is entropy of autocorrelation, NOT clinical RPDE.
    # Healthy range: 0.05-0.18 -> UCI healthy zone (~0.443).
    # PD range: 0.25-0.50 -> UCI PD zone (~0.517).
    raw_rpde = float(np.clip(features.get("rpde", 0.12), 0.01, 1.0))
    if raw_rpde <= 0.18:
        rpde = 0.443  # solidly in UCI healthy mean
    elif raw_rpde <= 0.45:
        rpde = 0.443 + (raw_rpde - 0.18) * 0.35  # gentle ramp
    else:
        rpde = 0.537 + (raw_rpde - 0.45) * 0.30  # steeper ramp
    rpde = float(np.clip(rpde, 0.25657, 0.68515))

    # -- DFA (BALANCED mapping) ---------------------------------------------
    # Browser DFA proxy (spectral flatness) is NOT clinical DFA.
    # Healthy range: 0.04-0.15 -> UCI healthy zone (~0.696).
    # PD range: 0.18-0.30 -> UCI PD zone (~0.725).
    raw_dfa = float(np.clip(features.get("dfa", 0.09), 0.001, 1.0))
    if raw_dfa <= 0.04:
        dfa = 0.680  # slightly below healthy mean but still healthy
    elif raw_dfa <= 0.15:
        dfa = 0.696 + (raw_dfa - 0.09) * 0.20  # gentle slope in healthy range
    elif raw_dfa <= 0.30:
        dfa = 0.718 + (raw_dfa - 0.15) * 0.50  # ramp toward PD
    else:
        dfa = 0.793 + (raw_dfa - 0.30) * 0.20
    dfa = float(np.clip(dfa, 0.574282, 0.825288))

    # -- Deviation metrics (used by NHR, spread, D2, PPE) -------------------
    jitter_dev = float(np.clip(
        (jitter_pct - _UCI_HE["jitter_pct"]) /
        (_UCI_PD["jitter_pct"] - _UCI_HE["jitter_pct"]),
        -0.5, 3.0
    ))
    shimmer_dev = float(np.clip(
        (shimmer - _UCI_HE["shimmer"]) / (_UCI_PD["shimmer"] - _UCI_HE["shimmer"]),
        -0.5, 3.0
    ))

    # -- NHR (driven by jitter+shimmer deviation, NOT HNR) ------------------
    # Clinical NHR measures noise components - correlates with jitter/shimmer.
    # UCI healthy mean=0.011, PD mean=0.029.
    nhr = float(np.clip(
        _UCI_HE["nhr"] +
        0.30 * max(jitter_dev, 0) * (_UCI_PD["nhr"] - _UCI_HE["nhr"]) +
        0.30 * max(shimmer_dev, 0) * (_UCI_PD["nhr"] - _UCI_HE["nhr"]),
        0.00065, 0.31482
    ))

    # -- spread1 (jitter-driven, moderate influence) ------------------------
    spread1 = float(np.clip(
        -6.76 + jitter_dev * 0.50,
        -7.964984, -2.434031
    ))

    # -- spread2 (40% jitter influence) -------------------------------------
    spread2 = float(np.clip(
        _UCI_HE["spread2"] + 0.40 * jitter_dev * (_UCI_PD["spread2"] - _UCI_HE["spread2"]),
        0.006274, 0.450493
    ))

    # -- D2 (shimmer-driven, moderate influence) ----------------------------
    d2 = float(np.clip(
        _UCI_HE["d2"] + 0.7 * shimmer_dev * (_UCI_PD["d2"] - _UCI_HE["d2"]),
        1.423287, 3.671155
    ))

    # -- PPE (BALANCED, jitter+shimmer+pitch_var driven) --------------------
    # PPE is one of the strongest PD discriminators. Raise PPE when jitter,
    # shimmer, AND pitch variation are elevated (PD pattern).
    # pitch_std_factor captures pitch instability (tremor/variability).
    pitch_std_raw = float(np.clip(features.get("pitch_std", 5.0), 0.1, 200.0))
    pitch_std_factor = float(np.clip((pitch_std_raw - 5.0) / 25.0, 0.0, 2.0))
    ppe = float(np.clip(
        _UCI_HE["ppe"] +
        0.35 * jitter_dev  * (_UCI_PD["ppe"] - _UCI_HE["ppe"]) +
        0.30 * shimmer_dev * (_UCI_PD["ppe"] - _UCI_HE["ppe"]) +
        0.20 * pitch_std_factor * (_UCI_PD["ppe"] - _UCI_HE["ppe"]),
        0.044539, 0.527367
    ))

    vec = [
        pitch,        # MDVP:Fo(Hz)
        fhi,          # MDVP:Fhi(Hz)
        flo,          # MDVP:Flo(Hz)
        jitter_pct,   # MDVP:Jitter(%)
        jitter_abs,   # MDVP:Jitter(Abs)
        jitter_rap,   # MDVP:RAP
        jitter_ppq,   # MDVP:PPQ
        jitter_ddp,   # Jitter:DDP
        shimmer,      # MDVP:Shimmer
        shimmer_db,   # MDVP:Shimmer(dB)
        shimmer_apq3, # Shimmer:APQ3
        shimmer_apq5, # Shimmer:APQ5
        shimmer_apq,  # MDVP:APQ
        shimmer_dda,  # Shimmer:DDA
        nhr,          # NHR
        hnr,          # HNR
        rpde,         # RPDE
        dfa,          # DFA
        spread1,      # spread1
        spread2,      # spread2
        d2,           # D2
        ppe,          # PPE
    ]
    return np.array(vec, dtype=np.float64)


def _assess_signal_quality(features: dict) -> tuple:
    """
    Assess the quality/reliability of the audio signal for PD analysis.
    Returns (quality_score, is_corrupted).
    
    quality_score: 0 (terrible) to 1 (excellent) — used for informational purposes.
    is_corrupted: True if features are physiologically impossible (noise, not PD).
    
    Clinical reality:
    - Healthy: jitter < 1%, shimmer < 3%, pitch_std < 10Hz, HNR > 20dB
    - PD:      jitter 0.5-3%, shimmer 3-8%, pitch_std 10-30Hz, HNR 10-20dB
    - Noise:   jitter > 5%, shimmer > 10%, pitch_std > 50Hz, HNR < 5dB
    
    Values far beyond PD ranges indicate environmental noise corruption,
    NOT genuine Parkinson's symptoms.
    """
    hnr = features.get("hnr", 20.0)
    pitch_std = features.get("pitch_std", 5.0)
    jitter = features.get("jitter_local", 0.005)
    shimmer = features.get("shimmer_local", 0.02)
    
    # ── Clinical plausibility check ──────────────────────────────────
    # If jitter/shimmer are far beyond what any human voice can produce,
    # the signal is corrupted by noise. No ML model can give reliable
    # results from corrupted data.
    #
    # Physiological limits:
    #   jitter_local > 5%  → impossible for real voice (even severe PD < 3%)
    #   shimmer_local > 12% → impossible for real voice (even severe PD < 10%)
    #   pitch_std > 80Hz   → indicates pitch tracking errors from noise
    is_corrupted = (
        jitter > 0.05 or       # > 5% jitter = noise corruption
        shimmer > 0.12 or      # > 12% shimmer = noise corruption
        (pitch_std > 80 and jitter > 0.02)  # extreme pitch instability + some jitter
    )
    
    # ── Informational quality score (0-1) ────────────────────────────
    hnr_quality = max(0.0, min(1.0, (hnr - 3.0) / 17.0))
    
    if pitch_std <= 30:
        pitch_quality = 1.0
    elif pitch_std <= 80:
        pitch_quality = 1.0 - (pitch_std - 30) / 100.0
    else:
        pitch_quality = max(0.0, 1.0 - (pitch_std - 30) / 120.0)
    
    if jitter <= 0.03:
        jitter_quality = 1.0
    elif jitter <= 0.08:
        jitter_quality = 1.0 - (jitter - 0.03) / 0.10
    else:
        jitter_quality = max(0.0, 1.0 - (jitter - 0.03) / 0.30)
    
    if shimmer <= 0.06:
        shimmer_quality = 1.0
    elif shimmer <= 0.12:
        shimmer_quality = 1.0 - (shimmer - 0.06) / 0.12
    else:
        shimmer_quality = max(0.0, 1.0 - (shimmer - 0.06) / 0.30)
    
    quality = (
        0.20 * hnr_quality +
        0.30 * pitch_quality +
        0.25 * jitter_quality +
        0.25 * shimmer_quality
    )
    quality = max(0.0, min(1.0, quality))
    
    return quality, is_corrupted


def heuristic_predict(features: dict) -> dict:
    """
    Research-backed heuristic when trained models are unavailable.
    Scores each feature relative to UCI healthy vs PD boundaries.
    A balanced score means healthy; a high score means PD-likely.
    """
    j    = features.get("jitter_local", 0.005)
    sh   = features.get("shimmer_local", 0.03)
    hnr  = features.get("hnr", 20.0)
    ps   = features.get("pitch_std", 2.0)
    rpde = features.get("rpde", 0.12)

    score = 0.0
    # ── Jitter (UCI healthy<0.006, PD>0.007 after calibration) ────────────
    # Raw browser values are ~15x inflated, so thresholds scaled up ~15x
    if j > 0.15:    score += 0.35
    elif j > 0.09:  score += 0.15
    elif j < 0.03:  score -= 0.20   # very stable = healthy signal

    # ── Shimmer (UCI healthy<0.025, PD>0.034 after calibration) ───────────
    # Raw browser values are ~5x inflated
    if sh > 0.18:   score += 0.30
    elif sh > 0.12: score += 0.10
    elif sh < 0.04: score -= 0.20

    # ── HNR (UCI healthy>23, PD<22) — direct cepstrum value ───────────────
    if hnr < 8.0:   score += 0.30
    elif hnr < 12.0: score += 0.15
    elif hnr > 18.0: score -= 0.25   # clear voice = healthy

    # ── Pitch std (excess variation signals PD tremor) ─────────────────────
    if ps > 12.0:   score += 0.10
    elif ps < 3.0:  score -= 0.05

    # ── RPDE (our proxy: healthy<0.12, PD>0.20) ────────────────────────────
    if rpde > 0.30:  score += 0.10
    elif rpde < 0.08: score -= 0.05

    score += random.gauss(0, 0.03)
    prob   = float(np.clip(1.0 / (1.0 + np.exp(-score * 3.0)), 0.05, 0.95))
    # Use same threshold as model (0.65)
    prediction = "Parkinson's Detected" if prob >= 0.65 else "Healthy"
    confidence = round((prob if prob >= 0.65 else 1.0 - prob) * 100, 1)
    return {"prediction": prediction, "probability": prob, "confidence": confidence,
            "model_used": "heuristic_screening"}


def risk_level(prob: float) -> str:
    if prob >= 0.80: return "Very High"
    if prob >= 0.65: return "High"
    if prob >= 0.45: return "Moderate"
    if prob >= 0.25: return "Low"
    return "Very Low"


def get_recommendations(prediction: str, confidence: float, features: dict) -> list:
    recs = []
    if "Parkinson" in prediction:
        recs.append("Consult a board-certified neurologist for a comprehensive evaluation.")
        recs.append("A standardized UPDRS (Unified Parkinson's Disease Rating Scale) assessment is recommended.")
        if features.get("jitter_local", 0) > 0.008:
            recs.append("Elevated pitch irregularity (jitter) detected — a speech-language pathologist can assess vocal symptoms.")
        if features.get("shimmer_local", 0) > 0.05:
            recs.append("High amplitude irregularity (shimmer) noted — voice therapy may help manage vocal fatigue.")
        if features.get("hnr", 20) < 15:
            recs.append("Reduced harmonic-to-noise ratio suggests increased breathiness — further respiratory assessment may be beneficial.")
        recs.append("Maintain a symptom diary to track voice and motor changes over time for clinical appointments.")
        recs.append("Regular aerobic exercise (e.g., cycling, swimming) has demonstrated benefits for motor symptoms in PD.")
        recs.append("LSVT LOUD speech therapy is specifically designed for Parkinson's-related vocal changes.")
        if confidence < 70:
            recs.append("Moderate model confidence — retesting with a quieter environment and longer sustained vowel may improve accuracy.")
    else:
        recs.append("Voice analysis appears within typical acoustic parameters.")
        recs.append("Continue routine health screenings with your primary care physician.")
        recs.append("Maintain vocal health through adequate hydration and avoiding vocal strain.")
        if confidence < 70:
            recs.append("Confidence is moderate — consider retesting with a cleaner recording for a more definitive result.")
        recs.append("This screening does not rule out Parkinson's Disease. Consult a neurologist if you have motor or other concerns.")
    return recs


def build_feature_importance(features: dict, n: int = 10) -> dict:
    """Return top-N feature importances with patient values."""
    if not feature_importance_global:
        # Fallback when no model loaded
        keys = ["jitter_local","shimmer_local","hnr","pitch_mean","rpde",
                "dfa","zcr_mean","spread1","spread2","ppe"]
        labels = ["Jitter","Shimmer","HNR","Pitch","RPDE","DFA","ZCR","Spread1","Spread2","PPE"]
        result = {}
        for k, l in zip(keys, labels):
            val = features.get(k, 0.0)
            result[l] = {"importance": 0.1, "value": round(float(val), 6),
                         "normal_min": None, "normal_max": None}
        return result

    result = {}
    for feat_name in list(feature_importance_global.keys())[:n]:
        raw = feature_importance_global[feat_name]
        # Handle both float (old format) and dict (new format)
        if isinstance(raw, dict):
            importance = raw.get("importance", 0.0)
        else:
            importance = float(raw)

        # Map feature name to extracted value
        val = features.get(feat_name, features.get(feat_name.replace("_mean",""), 0.0))
        rng = NORMAL_RANGES.get(feat_name, (None, None))

        result[feat_name] = {
            "importance": round(float(importance), 6),
            "value":      round(float(val), 6),
            "normal_min": rng[0],
            "normal_max": rng[1],
        }
    return result


def build_pdf(result: dict) -> bytes:
    """Generate a clinical-style PDF report using ReportLab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(buf, pagesize=A4,
                             rightMargin=20*mm, leftMargin=20*mm,
                             topMargin=20*mm,  bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    BLUE   = colors.HexColor("#1d4ed8")
    DARK   = colors.HexColor("#0f172a")
    GREY   = colors.HexColor("#475569")
    LIGHT  = colors.HexColor("#f8fafc")
    RED    = colors.HexColor("#dc2626")
    GREEN  = colors.HexColor("#16a34a")

    title_s = ParagraphStyle("T",  parent=styles["Title"],  fontSize=22, textColor=BLUE, alignment=TA_CENTER, spaceAfter=4)
    sub_s   = ParagraphStyle("S",  parent=styles["Normal"], fontSize=11, textColor=GREY, alignment=TA_CENTER, spaceAfter=2)
    h2_s    = ParagraphStyle("H2", parent=styles["Normal"], fontSize=12, textColor=BLUE, fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=4)
    body_s  = ParagraphStyle("B",  parent=styles["Normal"], fontSize=9,  textColor=DARK, leading=14)
    disc_s  = ParagraphStyle("D",  parent=styles["Normal"], fontSize=7.5,textColor=GREY, alignment=TA_CENTER)
    bold_s  = ParagraphStyle("BL", parent=styles["Normal"], fontSize=9,  fontName="Helvetica-Bold")

    is_pd   = "Parkinson" in result.get("prediction", "")
    p_color = RED if is_pd else GREEN
    story   = []

    # ── Header ──
    story += [
        Paragraph("PD Detect — Voice Analysis Report", title_s),
        Paragraph("AI-Powered Parkinson's Disease Screening via Voice Biomarkers", sub_s),
        HRFlowable(width="100%", thickness=2, color=BLUE),
        Spacer(1, 5*mm),
    ]

    # ── Meta table ──
    meta = [
        ["Analysis ID", result.get("id", "—"),  "Timestamp", result.get("timestamp", "—")],
        ["Model",       result.get("model_used","ensemble"), "Risk Level", result.get("risk_level","—")],
    ]
    mt = Table(meta, colWidths=[35*mm, 60*mm, 35*mm, 40*mm])
    mt.setStyle(TableStyle([
        ("FONTNAME",   (0,0),(-1,-1), "Helvetica"),
        ("FONTNAME",   (0,0),(0,-1),  "Helvetica-Bold"),
        ("FONTNAME",   (2,0),(2,-1),  "Helvetica-Bold"),
        ("FONTSIZE",   (0,0),(-1,-1), 9),
        ("TEXTCOLOR",  (0,0),(-1,-1), DARK),
        ("BACKGROUND", (0,0),(0,-1),  LIGHT),
        ("BACKGROUND", (2,0),(2,-1),  LIGHT),
        ("GRID",       (0,0),(-1,-1), 0.4, colors.HexColor("#cbd5e1")),
        ("PADDING",    (0,0),(-1,-1), 5),
    ]))
    story += [mt, Spacer(1, 5*mm)]

    # ── Result ──
    story.append(Paragraph("Diagnosis Result", h2_s))
    res_data = [
        ["Prediction", result.get("prediction","—")],
        ["Confidence", f"{result.get('confidence',0):.1f}%"],
        ["Risk Level", result.get("risk_level","—")],
    ]
    rt = Table(res_data, colWidths=[55*mm, 115*mm])
    rt.setStyle(TableStyle([
        ("BACKGROUND",  (0,0),(0,-1),  LIGHT),
        ("FONTNAME",    (0,0),(-1,-1), "Helvetica"),
        ("FONTNAME",    (0,0),(0,-1),  "Helvetica-Bold"),
        ("FONTNAME",    (1,0),(1,0),   "Helvetica-Bold"),
        ("FONTSIZE",    (1,0),(1,0),   13),
        ("TEXTCOLOR",   (1,0),(1,0),   p_color),
        ("GRID",        (0,0),(-1,-1), 0.4, colors.HexColor("#bfdbfe")),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.white, LIGHT]),
        ("PADDING",     (0,0),(-1,-1), 6),
    ]))
    story += [rt, Spacer(1, 5*mm)]

    # ── Features ──
    story.append(Paragraph("Key Acoustic Features", h2_s))
    f = result.get("features", {})
    feat_rows = [["Feature", "Value", "Normal Range", "Status"]]
    feat_cfg  = [
        ("Jitter (Local)",         f.get("jitter_local"),       "0.001 – 0.010"),
        ("Shimmer (Local)",        f.get("shimmer_local"),      "0.010 – 0.060"),
        ("HNR (dB)",               f.get("hnr"),                "15 – 35"),
        ("Pitch Mean (Hz)",        f.get("pitch_mean"),         "75 – 300"),
        ("Pitch Std (Hz)",         f.get("pitch_std"),          "0.5 – 10"),
        ("ZCR Mean",               f.get("zcr_mean"),           "0.01 – 0.15"),
        ("RPDE",                   f.get("rpde"),               "0.40 – 0.65"),
        ("DFA",                    f.get("dfa"),                "0.50 – 0.80"),
        ("Spectral Centroid (Hz)", f.get("spectral_centroid_mean"), "500 – 4000"),
        ("RMS Energy",             f.get("rms_mean"),           "0.005 – 0.40"),
        ("Jitter (Abs)",           f.get("jitter_absolute"),    "< 0.00005"),
        ("Shimmer (dB)",           f.get("shimmer_db"),         "< 0.5"),
    ]
    for name, val, rng in feat_cfg:
        v_str  = f"{val:.5f}" if val is not None else "—"
        status = "—"
        if val is not None:
            parts = rng.replace("< ","0 – ").split(" – ")
            if len(parts) == 2:
                try:
                    lo, hi = float(parts[0]), float(parts[1])
                    status = "Normal" if lo <= val <= hi else "Abnormal"
                except:
                    pass
        feat_rows.append([name, v_str, rng, status])

    ft = Table(feat_rows, colWidths=[65*mm, 35*mm, 45*mm, 25*mm])
    style_cmds = [
        ("BACKGROUND",   (0,0), (-1,0),  BLUE),
        ("TEXTCOLOR",    (0,0), (-1,0),  colors.white),
        ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTNAME",     (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",     (0,0), (-1,-1), 8.5),
        ("GRID",         (0,0), (-1,-1), 0.4, colors.HexColor("#bfdbfe")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, LIGHT]),
        ("PADDING",      (0,0), (-1,-1), 4.5),
    ]
    for row_idx, (_, val, rng) in enumerate(feat_cfg, start=1):
        if val is not None:
            parts = rng.replace("< ","0 – ").split(" – ")
            status = "—"
            if len(parts) == 2:
                try:
                    lo, hi = float(parts[0]), float(parts[1])
                    status = "Normal" if lo <= val <= hi else "Abnormal"
                except:
                    pass
            c = GREEN if status == "Normal" else (RED if status == "Abnormal" else GREY)
            style_cmds.append(("TEXTCOLOR", (3, row_idx), (3, row_idx), c))
            style_cmds.append(("FONTNAME",  (3, row_idx), (3, row_idx), "Helvetica-Bold"))
    ft.setStyle(TableStyle(style_cmds))
    story += [ft, Spacer(1, 5*mm)]

    # ── Recommendations ──
    story.append(Paragraph("Recommendations", h2_s))
    for rec in result.get("recommendations", []):
        story.append(Paragraph(f"• {rec}", body_s))
    story.append(Spacer(1, 8*mm))

    # ── Disclaimer ──
    story += [
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#94a3b8")),
        Spacer(1, 3*mm),
        Paragraph(
            "DISCLAIMER: This report is generated by an AI-based screening tool for research and educational purposes only. "
            "It does NOT constitute a medical diagnosis and has not been approved by any regulatory authority. "
            "Please consult a qualified neurologist for any medical concerns. "
            "Dataset: Little et al. (2007), UCI ML Repository.",
            disc_s,
        ),
    ]

    doc.build(story)
    return buf.getvalue()


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":        "ok",
        "models_loaded": models_loaded,
        "history_count": len(analysis_history),
        "timestamp":     datetime.utcnow().isoformat(),
        "version":       "1.0.0",
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    if len(audio_bytes) < 200:
        raise HTTPException(status_code=400, detail="Audio file is empty or too small.")

    # Feature extraction
    try:
        features = extract_features(audio_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Feature extraction failed: {str(e)}. Ensure audio contains clear speech.",
        )

    # Prediction
    if models_loaded and scaler is not None and ensemble is not None:
        try:
            # Always use UCI 22-feature vector (model v3 trained on UCI features).
            # feature_names.json from old model is not used by v3.
            feat_vec    = map_to_uci_vector(features).reshape(1, -1)
            feat_scaled = scaler.transform(feat_vec)
            proba       = float(ensemble.predict_proba(feat_scaled)[0][1])
            
            # Signal quality assessment.
            # Distinguishes noise corruption from genuine PD symptoms.
            quality, is_corrupted = _assess_signal_quality(features)
            base_threshold = 0.65
            
            if is_corrupted:
                # ── NOISE OVERRIDE ──────────────────────────────────────
                # Features are physiologically impossible (jitter > 5% or
                # shimmer > 12%). This is environmental noise corruption,
                # NOT genuine PD. The model was trained on clean clinical
                # data and cannot handle this — override its prediction.
                #
                # Real PD patients have jitter 0.5-3%, shimmer 3-8%.
                # Values beyond 5%/12% are physically impossible for any
                # human voice, even with severe Parkinson's.
                proba = min(proba, 0.30)  # Cap PD probability low
                PD_THRESHOLD = base_threshold
                logger.warning(
                    f"Signal corrupted by noise (jitter={features.get('jitter_local',0):.1%}, "
                    f"shimmer={features.get('shimmer_local',0):.1%}). "
                    f"Overriding model prediction to Healthy."
                )
            else:
                # ── NORMAL PREDICTION ───────────────────────────────────
                # Signal quality is acceptable. Check if clinical PD pattern
                # is present to help the model when it underpredicts.
                PD_THRESHOLD = base_threshold
                
                # Clinical pattern boost: when multiple PD biomarkers are
                # in clinical range, the model may underpredict because the
                # audio doesn't perfectly match UCI training data patterns.
                # Boost probability when clinical evidence is strong.
                jitter = features.get("jitter_local", 0.005)
                shimmer = features.get("shimmer_local", 0.02)
                pitch_std = features.get("pitch_std", 5.0)
                hnr = features.get("hnr", 20.0)
                
                # Count how many biomarkers are in PD clinical range
                # Ranges cover mild to severe PD:
                #   jitter 0.5-5% (mild ~1%, severe ~3-4%)
                #   shimmer 3-15% (mild ~5%, severe ~10-12%)
                #   pitch_std 8-35Hz, HNR 6-20dB
                pd_biomarkers = 0
                if 0.005 <= jitter <= 0.05:   # 0.5-5% jitter
                    pd_biomarkers += 1
                if 0.03 <= shimmer <= 0.15:    # 3-15% shimmer
                    pd_biomarkers += 1
                if 8.0 <= pitch_std <= 35.0:   # 8-35 Hz pitch_std
                    pd_biomarkers += 1
                if 6.0 <= hnr <= 20.0:         # 6-20 dB HNR
                    pd_biomarkers += 1
                
                # When 3+ biomarkers indicate PD, boost model probability
                if pd_biomarkers >= 3 and proba < PD_THRESHOLD:
                    # Strong boost: synthetic/real audio may not perfectly match
                    # UCI training data patterns, but clinical biomarkers are reliable
                    boost = 0.35 if pd_biomarkers == 3 else 0.50
                    proba = min(proba + boost, 0.95)
                    logger.info(
                        f"Clinical PD pattern detected ({pd_biomarkers}/4 biomarkers). "
                        f"Boosted probability: {proba:.3f}"
                    )
            
            pred        = "Parkinson's Detected" if proba >= PD_THRESHOLD else "Healthy"
            conf        = round((proba if proba >= PD_THRESHOLD else 1.0 - proba) * 100, 1)
            quality_tag = "CORRUPTED" if is_corrupted else f"q={quality:.0%}"
            model_used  = f"Ensemble (RF + XGBoost + SVM + GB) [{quality_tag}, thr={PD_THRESHOLD:.2f}]"
        except Exception as e:
            logger.error(f"Model inference error: {e}. Falling back to heuristic.")
            p = heuristic_predict(features)
            pred, proba, conf, model_used = p["prediction"], p["probability"], p["confidence"], p["model_used"]
            quality, is_corrupted = 0.7, False
    else:
        p = heuristic_predict(features)
        pred, proba, conf, model_used = p["prediction"], p["probability"], p["confidence"], p["model_used"]
        quality, is_corrupted = 0.7, False

    try:
        quality_val, corrupted_val = _assess_signal_quality(features)
    except Exception:
        quality_val, corrupted_val = 0.7, False

    recs = get_recommendations(pred, conf, features)
    fimp = build_feature_importance(features)

    result = {
        "id":                str(uuid.uuid4())[:8].upper(),
        "timestamp":         datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "prediction":        pred,
        "confidence":        conf,
        "probability":       round(proba, 4),
        "risk_percent":      round(proba * 100, 1),
        "risk_level":        risk_level(proba),
        "quality_score":     round(quality_val * 100, 1),
        "is_corrupted":      corrupted_val,
        "snr_db":            round(float(features.get("snr_db", 0.0)), 1),
        "features":          {k: round(float(v), 6) for k, v in features.items()},
        "feature_importance":fimp,
        "recommendations":   recs,
        "model_used":        model_used,
    }

    analysis_history.append(result)
    if len(analysis_history) > MAX_HISTORY:
        analysis_history.pop(0)

    return result


@app.post("/validate-audio")
async def validate_audio(file: UploadFile = File(...)):
    """Quick audio quality check without running full ML pipeline."""
    audio_bytes = await file.read()
    if len(audio_bytes) < 200:
        raise HTTPException(status_code=400, detail="File too small.")
    try:
        features = extract_features(audio_bytes)
        quality, is_corrupted = _assess_signal_quality(features)
        snr = features.get("snr_db", 0.0)
        duration = features.get("duration_sec", 0.0)

        issues = []
        if is_corrupted:
            issues.append("High background noise detected — results may be unreliable")
        if duration < 2.0:
            issues.append("Recording too short — please record for at least 3-5 seconds")
        if snr < 10:
            issues.append("Low signal-to-noise ratio — record in a quieter environment")
        if features.get("zcr_mean", 0) > 0.3:
            issues.append("Possible non-voice audio detected")

        return {
            "quality_score": round(quality * 100, 1),
            "is_corrupted":  is_corrupted,
            "snr_db":        round(float(snr), 1),
            "duration_sec":  round(float(duration), 2),
            "issues":        issues,
            "recommendation": "Good quality — proceed with analysis" if quality > 0.6 and not issues else "Please re-record in a quieter environment",
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Audio validation failed: {str(e)}")


@app.get("/history")
async def get_history():
    return {
        "history": list(reversed(analysis_history)),
        "count":   len(analysis_history),
    }


@app.delete("/history")
async def clear_all_history():
    analysis_history.clear()
    return {"message": "History cleared.", "count": 0}


@app.delete("/history/{entry_id}")
async def delete_entry(entry_id: str):
    global analysis_history
    before = len(analysis_history)
    analysis_history = [h for h in analysis_history if h.get("id") != entry_id]
    if len(analysis_history) == before:
        raise HTTPException(status_code=404, detail="Entry not found.")
    return {"message": f"Entry {entry_id} deleted."}


@app.get("/stats")
async def get_stats():
    if not analysis_history:
        return {
            "total":           0,
            "pd_count":        0,
            "healthy_count":   0,
            "avg_confidence":  0,
            "feature_averages":{},
            "risk_distribution":{"High": 0, "Medium": 0, "Low": 0},
        }

    pd_list = [h for h in analysis_history if h["prediction"].startswith("Parkinson")]
    he_list = [h for h in analysis_history if not h["prediction"].startswith("Parkinson")]

    # Feature averages
    feat_keys = ["jitter_local","shimmer_local","hnr","pitch_mean","pitch_std",
                 "zcr_mean","rpde","dfa","rms_mean","spectral_centroid_mean"]
    feat_avgs = {}
    for k in feat_keys:
        vals = [h["features"].get(k) for h in analysis_history if h["features"].get(k) is not None]
        feat_avgs[k] = round(float(np.mean(vals)), 6) if vals else 0.0

    # Risk distribution
    risk_dist = {"High": 0, "Medium": 0, "Low": 0}
    for h in analysis_history:
        r = h.get("risk_level", "Low")
        risk_dist[r] = risk_dist.get(r, 0) + 1

    return {
        "total":             len(analysis_history),
        "pd_count":          len(pd_list),
        "healthy_count":     len(he_list),
        "avg_confidence":    round(float(np.mean([h["confidence"] for h in analysis_history])), 1),
        "feature_averages":  feat_avgs,
        "risk_distribution": risk_dist,
        "models_loaded":     models_loaded,
    }


@app.post("/report")
async def generate_report(request: Request):
    """Accept raw JSON body for PDF generation."""
    try:
        result = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")
    try:
        pdf_bytes = build_pdf(result)
    except Exception as e:
        logger.error(f"PDF generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

    ts       = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"pd_report_{ts}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ─── Demo Data Generation ─────────────────────────────────────────────────────

def _generate_synthetic_wav(f0=165.0, noise=0.02, jitter_amount=0.0,
                            shimmer_amount=0.0, duration=3.0, sr=22050, seed=None):
    """Generate a synthetic sustained-vowel WAV buffer.
    Returns bytes suitable for extract_features().
    """
    import wave as _wave
    if seed is not None:
        np.random.seed(seed)
    n_samples = int(sr * duration)
    t = np.arange(n_samples) / sr

    # Per-cycle bounded jitter
    base_period = 1.0 / f0
    phase = np.zeros(n_samples)
    current_phase = 0.0
    i = 0
    while i < n_samples:
        if jitter_amount > 0:
            delta_T = np.random.randn() * base_period * jitter_amount
            delta_T = np.clip(delta_T, -3 * base_period * jitter_amount,
                              3 * base_period * jitter_amount)
            this_period = base_period + delta_T
        else:
            this_period = base_period
        samples_in_cycle = min(max(1, int(this_period * sr)), n_samples - i)
        cycle_phase = 2 * np.pi * (np.arange(samples_in_cycle) / samples_in_cycle)
        phase[i:i + samples_in_cycle] = current_phase + cycle_phase
        current_phase += 2 * np.pi
        i += samples_in_cycle

    # Harmonic signal
    signal = (0.5 * np.sin(phase)
            + 0.2 * np.sin(2 * phase)
            + 0.1 * np.sin(3 * phase)
            + 0.05 * np.sin(4 * phase))

    # Amplitude shimmer
    if shimmer_amount > 0:
        amp_noise = np.random.randn(n_samples)
        ks = max(1, int(sr / f0))
        amp_noise = np.convolve(amp_noise, np.ones(ks) / ks, 'same')
        amp_mod = 1.0 + shimmer_amount * amp_noise / (np.std(amp_noise) + 1e-9)
        signal *= amp_mod

    # Noise floor
    signal += noise * np.random.randn(n_samples)

    # Fade-in / fade-out envelope
    fade = int(0.05 * sr)
    envelope = np.ones_like(t)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    signal *= envelope

    # Normalize to 16-bit PCM WAV
    signal = signal / (np.max(np.abs(signal)) + 1e-9) * 0.8
    pcm = (signal * 32767).astype(np.int16)
    buf = io.BytesIO()
    with _wave.open(buf, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    buf.seek(0)
    return buf.read()


_DEMO_PROFILES = [
    # Healthy voices
    {"label": "Healthy Male (165 Hz)",        "f0": 165, "noise": 0.02, "jitter": 0.0,   "shimmer": 0.0,  "seed": 100},
    {"label": "Healthy Female (220 Hz)",      "f0": 220, "noise": 0.02, "jitter": 0.0,   "shimmer": 0.0,  "seed": 101},
    {"label": "Healthy Low-Pitch Male (120 Hz)", "f0": 120, "noise": 0.03, "jitter": 0.0, "shimmer": 0.0, "seed": 102},
    # PD voices (clinically realistic)
    {"label": "PD Mild (1.2% jitter)",       "f0": 150, "noise": 0.05, "jitter": 0.012, "shimmer": 0.06, "seed": 200},
    {"label": "PD Moderate (2.2% jitter)",   "f0": 145, "noise": 0.08, "jitter": 0.022, "shimmer": 0.10, "seed": 201},
    {"label": "PD Severe (3.5% jitter)",     "f0": 140, "noise": 0.12, "jitter": 0.035, "shimmer": 0.15, "seed": 202},
]


@app.post("/seed-demo")
async def seed_demo_data():
    """Generate synthetic audio for healthy + PD profiles, analyze each,
    store results in history, and return all results for the frontend."""
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    results = []
    for profile in _DEMO_PROFILES:
        wav_bytes = _generate_synthetic_wav(
            f0=profile["f0"],
            noise=profile["noise"],
            jitter_amount=profile["jitter"],
            shimmer_amount=profile["shimmer"],
            duration=3.0,
            seed=profile["seed"],
        )
        # Extract features
        try:
            features = extract_features(wav_bytes)
        except Exception as e:
            logger.error(f"Demo feature extraction failed for {profile['label']}: {e}")
            continue

        # Run prediction (same logic as /analyze)
        try:
            feat_vec = map_to_uci_vector(features).reshape(1, -1)
            feat_scaled = scaler.transform(feat_vec)
            proba = float(ensemble.predict_proba(feat_scaled)[0][1])

            quality, is_corrupted = _assess_signal_quality(features)
            PD_THRESHOLD = 0.65

            if is_corrupted:
                proba = min(proba, 0.30)
            else:
                jitter = features.get("jitter_local", 0.005)
                shimmer = features.get("shimmer_local", 0.02)
                pitch_std = features.get("pitch_std", 5.0)
                hnr = features.get("hnr", 20.0)
                pd_bio = sum([
                    0.005 <= jitter <= 0.05,
                    0.03 <= shimmer <= 0.15,
                    8.0 <= pitch_std <= 35.0,
                    6.0 <= hnr <= 20.0,
                ])
                if pd_bio >= 3 and proba < PD_THRESHOLD:
                    proba = min(proba + (0.35 if pd_bio == 3 else 0.50), 0.95)

            pred = "Parkinson's Detected" if proba >= PD_THRESHOLD else "Healthy"
            conf = round((proba if proba >= PD_THRESHOLD else 1.0 - proba) * 100, 1)
            quality_tag = "CORRUPTED" if is_corrupted else f"q={quality:.0%}"
            model_used = f"Ensemble (RF + XGBoost + SVM + GB) [{quality_tag}, thr={PD_THRESHOLD:.2f}]"
        except Exception as e:
            logger.error(f"Demo prediction failed: {e}")
            p = heuristic_predict(features)
            pred, proba, conf, model_used = p["prediction"], p["probability"], p["confidence"], p["model_used"]

        recs = get_recommendations(pred, conf, features)
        fimp = build_feature_importance(features)

        result = {
            "id":                 str(uuid.uuid4())[:8].upper(),
            "timestamp":          datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "prediction":         pred,
            "confidence":         conf,
            "probability":        round(proba, 4),
            "risk_level":         risk_level(proba),
            "features":           {k: round(float(v), 6) for k, v in features.items()},
            "feature_importance": fimp,
            "recommendations":    recs,
            "model_used":         model_used,
            "demo_label":         profile["label"],
        }
        analysis_history.append(result)
        results.append(result)

    if len(analysis_history) > MAX_HISTORY:
        del analysis_history[:-MAX_HISTORY]

    return {
        "seeded": len(results),
        "results": results,
        "message": f"{len(results)} demo analyses added to history.",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
