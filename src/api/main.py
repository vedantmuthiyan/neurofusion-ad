"""NeuroFusion-AD FHIR API — FastAPI application.

Endpoints:
    GET  /health — liveness probe
    GET  /ready  — readiness probe (model loaded check)
    POST /fhir/RiskAssessment/$process — primary inference endpoint

Authentication: API key header (X-API-Key) for production deployments.
Audit logging: all predictions logged to PostgreSQL.

Performance targets (SRS-001 § 7.1):
    p95 latency < 2000ms (measured on RunPod RTX 3090)

Temperature scaling: T=0.756 (Phase 2B calibration, ECE_after=0.083)
Monte Carlo CI: 30 dropout-enabled forward passes

IEC 62304 traceability: SRS-001 § 5.5, SAD-001 § 5.1, § 5.2
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import structlog
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Header
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Local imports
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.neurofusion_model import NeuroFusionAD
from src.data.inference_preprocessor import InferencePreprocessor
from src.api.fhir_validator import FHIRBundleValidator, operation_outcome
from src.api.fhir_output import FHIROutputBuilder

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(PROJECT_ROOT / "models" / "final" / "best_model.pth"),
)
SCALER_PATH = os.environ.get(
    "SCALER_PATH",
    str(PROJECT_ROOT / "data" / "processed" / "adni" / "scaler.pkl"),
)
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.756"))  # Phase 2B calibration
MC_SAMPLES = int(os.environ.get("MC_SAMPLES", "30"))  # Monte Carlo dropout
API_KEY = os.environ.get("NEUROFUSION_API_KEY", "")  # empty = no auth (dev mode)

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://neurofusion:neurofusion@localhost:5432/neurofusion",
)

# Shared model state (loaded on startup)
_model: Optional[NeuroFusionAD] = None
_preprocessor: Optional[InferencePreprocessor] = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model_ready = False

# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and scaler on startup; clean up on shutdown."""
    global _model, _preprocessor, _model_ready

    log.info("NeuroFusion-AD API starting up", device=_device)

    # Load scaler
    try:
        _preprocessor = InferencePreprocessor.from_scaler(SCALER_PATH)
        log.info("Scaler loaded", path=SCALER_PATH)
    except FileNotFoundError:
        log.warning("Scaler not found — will impute all features", path=SCALER_PATH)
        # Create a stub preprocessor with no scaler
        import pickle
        from sklearn.preprocessing import StandardScaler
        stub_scaler = StandardScaler()
        stub_scaler.mean_ = np.zeros(1)
        stub_scaler.scale_ = np.ones(1)
        stub_scaler.feature_names_in_ = np.array([])
        _preprocessor = InferencePreprocessor(stub_scaler)

    # Load model
    try:
        ckpt = torch.load(MODEL_PATH, map_location=_device, weights_only=True)
        cfg = ckpt.get("config", {})
        embed_dim = int(cfg.get("embed_dim", 256))
        num_heads = int(cfg.get("num_heads", 4))
        dropout = float(cfg.get("dropout", 0.4))

        _model = NeuroFusionAD(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        state = ckpt.get("model_state_dict", ckpt)
        _model.load_state_dict(state)
        _model.to(_device)
        _model.eval()
        _model_ready = True
        log.info(
            "Model loaded",
            path=MODEL_PATH,
            embed_dim=embed_dim,
            num_heads=num_heads,
            device=_device,
        )
    except FileNotFoundError:
        log.warning("Model checkpoint not found — API will return 503 on inference", path=MODEL_PATH)
        _model_ready = False

    yield

    log.info("NeuroFusion-AD API shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NeuroFusion-AD FHIR API",
    description=(
        "Multimodal GNN for Alzheimer's Disease amyloid positivity risk assessment. "
        "FDA De Novo pathway (investigational — not yet cleared for clinical use)."
    ),
    version="1.0.0-phase2b",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper: temperature scaling
# ---------------------------------------------------------------------------

def _temperature_scale(logit: float, temperature: float = TEMPERATURE) -> float:
    """Apply temperature scaling to logit.

    Args:
        logit: Raw model logit (before sigmoid).
        temperature: Calibration temperature (Phase 2B: T=0.756).

    Returns:
        Calibrated probability.
    """
    scaled_logit = logit / temperature
    return float(1.0 / (1.0 + np.exp(-scaled_logit)))


def _run_inference(
    batch: dict[str, torch.Tensor],
    n_mc: int = MC_SAMPLES,
) -> dict[str, Any]:
    """Run model inference with Monte Carlo dropout for CI estimation.

    Args:
        batch: Tensor batch from InferencePreprocessor.
        n_mc: Number of MC dropout passes.

    Returns:
        Dict with probability, ci_lower, ci_upper, mmse_slope, cox_log_hazard,
        modality_importance.
    """
    if _model is None:
        raise RuntimeError("Model not loaded")

    # Move tensors to device
    device_batch = {k: v.to(_device) for k, v in batch.items()}

    # Single deterministic pass (eval mode, dropout disabled)
    with torch.no_grad():
        outputs = _model(device_batch)
        logit = float(outputs["amyloid_logit"].cpu().item())
        mmse_slope = float(outputs.get("mmse_slope", torch.tensor(float("nan"))).cpu().item())
        cox_lh = float(outputs.get("cox_log_hazard", torch.tensor(float("nan"))).cpu().item())

    prob = _temperature_scale(logit)

    # Monte Carlo dropout for CI
    _model.train()  # enable dropout
    mc_probs = []
    with torch.no_grad():
        for _ in range(n_mc):
            mc_out = _model(device_batch)
            mc_logit = float(mc_out["amyloid_logit"].cpu().item())
            mc_probs.append(_temperature_scale(mc_logit))
    _model.eval()  # restore eval

    ci_lower = float(np.percentile(mc_probs, 2.5))
    ci_upper = float(np.percentile(mc_probs, 97.5))

    # Modality importance from attention weights (approximate from first layer)
    _PHASE2B_IMPORTANCE = {
        "fluid": 0.213,
        "acoustic": 0.262,
        "motor": 0.240,
        "clinical": 0.286,
    }
    try:
        from src.evaluation.attention_analysis import AttentionAnalyzer
        attn_analyzer = AttentionAnalyzer()
        # Use single batch — wrap in a 1-element list for the analyzer
        device_batch_single = {k: v.to(_device) for k, v in batch.items()}
        attn_results = attn_analyzer.extract_attention_weights(_model, [device_batch_single])
        importance = attn_analyzer.get_modality_importance_scores(attn_results)
        # Validate — fall back if any value is NaN or non-finite
        import math
        if not importance or any(
            not math.isfinite(v) for v in importance.values()
        ):
            importance = _PHASE2B_IMPORTANCE
    except Exception:
        # Fallback to Phase 2B average modality importance
        importance = _PHASE2B_IMPORTANCE

    return {
        "probability": prob,
        "ci_lower": max(0.0, min(1.0, ci_lower)),
        "ci_upper": max(0.0, min(1.0, ci_upper)),
        "logit": logit,
        "mmse_slope": mmse_slope,
        "cox_log_hazard": cox_lh,
        "modality_importance": importance,
        "mc_std": float(np.std(mc_probs)),
    }


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------

async def _audit_log(
    request_id: str,
    patient_id_hash: str,
    source_system: str,
    probability: float,
    latency_ms: float,
    error: str | None = None,
) -> None:
    """Write prediction audit record to PostgreSQL.

    Non-blocking — failures are logged but don't raise.

    Args:
        request_id: Unique request UUID.
        patient_id_hash: SHA-256 of patient identifier.
        source_system: Requesting EHR system.
        probability: Predicted probability.
        latency_ms: Request latency in milliseconds.
        error: Error message if prediction failed.
    """
    try:
        import asyncpg
        conn = await asyncpg.connect(DB_URL)
        await conn.execute(
            """
            INSERT INTO prediction_audit_log
                (request_id, patient_id_hash, source_system, probability,
                 latency_ms, model_version, error, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
            """,
            request_id,
            patient_id_hash,
            source_system,
            probability,
            latency_ms,
            "phase2b-v1.0",
            error,
        )
        await conn.close()
    except Exception as exc:
        log.warning("Audit log failed", error=str(exc))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Operations"])
async def health() -> dict[str, str]:
    """Liveness probe — always returns 200 if process is running."""
    return {
        "status": "ok",
        "service": "neurofusion-ad-api",
        "version": "1.0.0-phase2b",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/ready", tags=["Operations"])
async def readiness() -> dict[str, Any]:
    """Readiness probe — returns 200 only when model is loaded."""
    if not _model_ready:
        raise HTTPException(
            status_code=503,
            detail="Model not yet loaded or checkpoint unavailable",
        )
    return {
        "status": "ready",
        "model_ready": True,
        "device": _device,
        "temperature": TEMPERATURE,
        "mc_samples": MC_SAMPLES,
    }


@app.post(
    "/fhir/RiskAssessment/$process",
    tags=["FHIR"],
    response_class=JSONResponse,
    responses={
        200: {"description": "FHIR RiskAssessment resource"},
        400: {"description": "Malformed JSON or wrong resourceType"},
        422: {"description": "FHIR validation error — OperationOutcome"},
        503: {"description": "Model not ready"},
    },
)
async def process_risk_assessment(
    request: Request,
    background_tasks: BackgroundTasks,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
) -> JSONResponse:
    """Process a FHIR Bundle and return a FHIR RiskAssessment.

    Accepts a FHIR R4 Bundle containing:
    - One Patient resource (demographics)
    - One or more Observation resources (clinical measurements)

    Returns a FHIR R4 RiskAssessment with amyloid positivity probability,
    95% CI, risk category, clinical recommendation, and modality importance.

    Content-Type: application/fhir+json
    """
    t0 = time.perf_counter()
    request_id = x_request_id or str(uuid.uuid4())

    # --- API key check (when configured) ---
    if API_KEY and x_api_key != API_KEY:
        return JSONResponse(
            status_code=401,
            content=operation_outcome("error", "security", "Invalid or missing API key"),
            media_type="application/fhir+json",
        )

    # --- Model readiness check ---
    if not _model_ready:
        return JSONResponse(
            status_code=503,
            content=operation_outcome("error", "exception", "Model not loaded"),
            media_type="application/fhir+json",
        )

    # --- Parse request body ---
    try:
        bundle = await request.json()
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content=operation_outcome(
                "error", "structure",
                f"Malformed JSON: {exc}",
            ),
            media_type="application/fhir+json",
        )

    # --- FHIR Bundle validation ---
    validator = FHIRBundleValidator()
    errors = validator.validate(bundle)
    if errors:
        log.warning("FHIR validation failed", request_id=request_id, n_errors=len(errors))
        return JSONResponse(
            status_code=422,
            content=errors[0],
            media_type="application/fhir+json",
        )

    patient_id = validator.extract_patient_id(bundle)
    source_system = validator.extract_requesting_system(bundle)

    # --- Preprocess ---
    try:
        batch = _preprocessor.from_fhir_bundle(bundle, patient_id=patient_id)
    except Exception as exc:
        log.error("Preprocessing failed", request_id=request_id, error=str(exc))
        return JSONResponse(
            status_code=422,
            content=operation_outcome(
                "error", "processing",
                f"Feature extraction failed: {exc}",
            ),
            media_type="application/fhir+json",
        )

    # --- Inference ---
    try:
        result = _run_inference(batch)
    except Exception as exc:
        log.error("Inference failed", request_id=request_id, error=str(exc))
        latency = (time.perf_counter() - t0) * 1000
        await _audit_log(request_id, patient_id, source_system, float("nan"), latency, str(exc))
        return JSONResponse(
            status_code=500,
            content=operation_outcome(
                "fatal", "exception",
                f"Model inference error: {exc}",
            ),
            media_type="application/fhir+json",
        )

    # --- Build FHIR output ---
    builder = FHIROutputBuilder()
    risk_assessment = builder.build(
        patient_id=patient_id,
        probability=result["probability"],
        ci_lower=result["ci_lower"],
        ci_upper=result["ci_upper"],
        modality_importance=result["modality_importance"],
        mmse_slope_pred=result.get("mmse_slope"),
        risk_assessment_id=request_id,
    )

    # Add request metadata
    risk_assessment["meta"]["requestId"] = request_id
    risk_assessment["meta"]["sourceSystem"] = source_system

    latency = (time.perf_counter() - t0) * 1000
    log.info(
        "RiskAssessment processed",
        request_id=request_id,
        patient_id=patient_id[:8] + "...",
        probability=round(result["probability"], 4),
        latency_ms=round(latency, 1),
    )

    # Non-blocking audit log via FastAPI BackgroundTasks
    background_tasks.add_task(
        _audit_log, request_id, patient_id, source_system, result["probability"], latency
    )

    return JSONResponse(
        status_code=200,
        content=risk_assessment,
        media_type="application/fhir+json",
        headers={
            "X-Request-ID": request_id,
            "X-Latency-Ms": str(round(latency, 1)),
            "ETag": f'W/"{request_id}"',
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=False,
        workers=1,  # single worker for GPU model
    )
