"""NeuroFusion-AD Clinical Demo Backend.

Serves 3 pre-defined patient scenarios and proxies inference requests
to the live NeuroFusion-AD FHIR API at localhost:8000.

Scenarios:
    1. Margaret Chen  — 77yo F, HIGH risk (pTau217 elevated, MMSE declining)
    2. Robert Martinez — 65yo M, MODERATE risk (borderline biomarkers)
    3. Dorothy Walsh   — 82yo F, MODERATE/LOW risk (high age but stable MMSE)

Run:
    python demo/backend/demo_api.py
    → Serves frontend at http://localhost:3000
    → API at http://localhost:3000/api/...
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import structlog
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

log = structlog.get_logger(__name__)

FHIR_API_URL = os.environ.get("FHIR_API_URL", "http://localhost:8000")
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

app = FastAPI(title="NeuroFusion-AD Demo", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, dict[str, Any]] = {
    "margaret-chen": {
        "id": "margaret-chen",
        "name": "Margaret Chen",
        "age": 77,
        "sex": "F",
        "education_years": 16,
        "apoe4_count": 1,
        "mmse_baseline": 22.0,
        "ptau217": 0.85,    # pg/mL — elevated (threshold ~0.4)
        "nfl_plasma": 18.5, # pg/mL — borderline elevated
        "tau_csf": 420.0,   # pg/mL — elevated
        "narrative": (
            "77-year-old female presenting with subjective memory complaints over 18 months. "
            "MMSE 22/30 (declined from 27 two years ago). Elevated plasma pTau217. "
            "APOE ε3/ε4. No prior amyloid imaging."
        ),
        "bundle": _make_bundle("margaret-chen-001", "female", "1948-05-12", [
            ("72107-6",  "MMSE total score",   22.0, "score"),
            ("100025-0", "Plasma pTau217",      0.85, "pg/mL"),
            ("81600-4",  "NfL plasma",         18.5, "pg/mL"),
            ("14683-7",  "Total tau CSF",      420.0, "pg/mL"),
            ("30155-3",  "APOE4 genotype",      1.0, "{copies}"),
        ]) if False else None,  # built below via _build_scenario_bundles()
    },
    "robert-martinez": {
        "id": "robert-martinez",
        "name": "Robert Martinez",
        "age": 65,
        "sex": "M",
        "education_years": 14,
        "apoe4_count": 0,
        "mmse_baseline": 26.0,
        "ptau217": 0.45,
        "nfl_plasma": 12.1,
        "tau_csf": 180.0,
        "narrative": (
            "65-year-old male, annual cognitive screening. Mild word-finding difficulties "
            "reported by spouse. MMSE 26/30 (stable over 12 months). Borderline pTau217. "
            "APOE ε3/ε3. No family history of dementia."
        ),
        "bundle": None,
    },
    "dorothy-walsh": {
        "id": "dorothy-walsh",
        "name": "Dorothy Walsh",
        "age": 82,
        "sex": "F",
        "education_years": 12,
        "apoe4_count": 2,
        "mmse_baseline": 24.0,
        "ptau217": 0.32,
        "nfl_plasma": 15.0,
        "tau_csf": 195.0,
        "narrative": (
            "82-year-old female, routine memory clinic follow-up. MMSE 24/30 (stable × 3 years). "
            "Age-related biomarker elevation but pTau217 below threshold. "
            "APOE ε4/ε4 (homozygous) — high genetic risk but biomarker-negative pattern."
        ),
        "bundle": None,
    },
}


def _make_bundle(
    patient_id: str,
    gender: str,
    birth_date: str,
    observations: list[tuple[str, str, float, str]],
) -> dict[str, Any]:
    """Build a minimal FHIR Bundle for a demo scenario."""
    entries: list[dict] = [
        {
            "resource": {
                "resourceType": "Patient",
                "id": patient_id,
                "gender": gender,
                "birthDate": birth_date,
            }
        }
    ]
    for loinc_code, display, value, unit in observations:
        entries.append({
            "resource": {
                "resourceType": "Observation",
                "status": "final",
                "code": {
                    "coding": [{"system": "http://loinc.org", "code": loinc_code, "display": display}]
                },
                "subject": {"reference": f"Patient/{patient_id}"},
                "valueQuantity": {"value": value, "unit": unit},
            }
        })
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "meta": {"source": "https://demo.neurofusion-ad.example.com"},
        "entry": entries,
    }


# Build bundles now that helper is defined
_BUNDLES_SPEC: dict[str, tuple] = {
    "margaret-chen": (
        "margaret-chen-001", "female", "1948-05-12",
        [
            ("72107-6",  "MMSE total score",  22.0, "score"),
            ("100025-0", "Plasma pTau217",     0.85, "pg/mL"),
            ("81600-4",  "NfL plasma",        18.5, "pg/mL"),
            ("14683-7",  "Total tau CSF",     420.0, "pg/mL"),
            ("30155-3",  "APOE4 genotype",     1.0, "{copies}"),
        ],
    ),
    "robert-martinez": (
        "robert-martinez-001", "male", "1960-11-03",
        [
            ("72107-6",  "MMSE total score",  26.0, "score"),
            ("100025-0", "Plasma pTau217",     0.45, "pg/mL"),
            ("81600-4",  "NfL plasma",        12.1, "pg/mL"),
            ("14683-7",  "Total tau CSF",     180.0, "pg/mL"),
            ("30155-3",  "APOE4 genotype",     0.0, "{copies}"),
        ],
    ),
    "dorothy-walsh": (
        "dorothy-walsh-001", "female", "1943-08-19",
        [
            ("72107-6",  "MMSE total score",  24.0, "score"),
            ("100025-0", "Plasma pTau217",     0.32, "pg/mL"),
            ("81600-4",  "NfL plasma",        15.0, "pg/mL"),
            ("14683-7",  "Total tau CSF",     195.0, "pg/mL"),
            ("30155-3",  "APOE4 genotype",     2.0, "{copies}"),
        ],
    ),
}

for _sid, _spec in _BUNDLES_SPEC.items():
    SCENARIOS[_sid]["bundle"] = _make_bundle(*_spec)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/scenarios")
async def list_scenarios() -> JSONResponse:
    """Return summary list of all demo scenarios (no results)."""
    summaries = [
        {
            "id": s["id"],
            "name": s["name"],
            "age": s["age"],
            "sex": s["sex"],
            "mmse": s["mmse_baseline"],
            "apoe4": s["apoe4_count"],
            "narrative": s["narrative"],
        }
        for s in SCENARIOS.values()
    ]
    return JSONResponse(content=summaries)


@app.post("/api/predict/{scenario_id}")
async def predict_scenario(scenario_id: str) -> JSONResponse:
    """Run inference for a demo scenario via the live FHIR API."""
    if scenario_id not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f"Scenario '{scenario_id}' not found")

    scenario = SCENARIOS[scenario_id]
    bundle = scenario["bundle"]

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{FHIR_API_URL}/fhir/RiskAssessment/$process",
                json=bundle,
                headers={"Content-Type": "application/fhir+json"},
            )
        if resp.status_code != 200:
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"FHIR API error: {resp.text[:500]}",
            )
        risk_assessment = resp.json()
    except httpx.ConnectError:
        # Fall back to pre-computed results if API is not available
        log.warning("FHIR API unavailable — using fallback results", scenario_id=scenario_id)
        risk_assessment = _fallback_result(scenario_id)

    # Enrich with scenario metadata for the frontend
    return JSONResponse(content={
        "scenario": {
            "id": scenario["id"],
            "name": scenario["name"],
            "age": scenario["age"],
            "sex": scenario["sex"],
            "apoe4_count": scenario["apoe4_count"],
            "mmse_baseline": scenario["mmse_baseline"],
            "ptau217": scenario["ptau217"],
            "nfl_plasma": scenario["nfl_plasma"],
            "narrative": scenario["narrative"],
        },
        "fhir": risk_assessment,
        "survival_curve": _generate_survival_curve(scenario_id, risk_assessment),
    })


def _fallback_result(scenario_id: str) -> dict[str, Any]:
    """Pre-computed fallback results when FHIR API is unavailable."""
    fallbacks = {
        "margaret-chen": {
            "resourceType": "RiskAssessment",
            "status": "final",
            "subject": {"reference": "Patient/margaret-chen-001"},
            "prediction": [{
                "outcome": {"coding": [{"code": "413448000"}], "text": "Amyloid positivity risk: HIGH"},
                "probabilityDecimal": 0.9422,
                "qualitativeRisk": {"coding": [{"code": "high", "display": "High risk"}]},
                "whenRange": {"low": {"value": 0.562}, "high": {"value": 0.987}},
                "rationale": "Phase 2B model output (fallback — FHIR API unavailable)",
            }],
            "note": [{"text": (
                "NeuroFusion-AD predicts 94.2% probability of amyloid positivity (HIGH risk). "
                "RECOMMENDATION: Consider confirmatory amyloid PET or CSF testing. "
                "NOTE: Investigational SaMD — not for clinical use."
            )}],
            "extension": [{"url": "...modality-importance", "extension": [
                {"url": "fluid", "valueDecimal": 0.342},
                {"url": "acoustic", "valueDecimal": 0.198},
                {"url": "motor", "valueDecimal": 0.176},
                {"url": "clinical", "valueDecimal": 0.284},
            ]}],
        },
        "robert-martinez": {
            "resourceType": "RiskAssessment",
            "status": "final",
            "subject": {"reference": "Patient/robert-martinez-001"},
            "prediction": [{
                "outcome": {"coding": [{"code": "723509005"}], "text": "Amyloid positivity risk: MODERATE"},
                "probabilityDecimal": 0.5241,
                "qualitativeRisk": {"coding": [{"code": "moderate", "display": "Moderate risk"}]},
                "whenRange": {"low": {"value": 0.381}, "high": {"value": 0.667}},
                "rationale": "Phase 2B model output (fallback)",
            }],
            "note": [{"text": (
                "NeuroFusion-AD predicts 52.4% probability of amyloid positivity (MODERATE risk). "
                "RECOMMENDATION: Monitor per standard of care. "
                "NOTE: Investigational SaMD — not for clinical use."
            )}],
            "extension": [{"url": "...modality-importance", "extension": [
                {"url": "fluid", "valueDecimal": 0.213},
                {"url": "acoustic", "valueDecimal": 0.262},
                {"url": "motor", "valueDecimal": 0.240},
                {"url": "clinical", "valueDecimal": 0.286},
            ]}],
        },
        "dorothy-walsh": {
            "resourceType": "RiskAssessment",
            "status": "final",
            "subject": {"reference": "Patient/dorothy-walsh-001"},
            "prediction": [{
                "outcome": {"coding": [{"code": "723509005"}], "text": "Amyloid positivity risk: MODERATE"},
                "probabilityDecimal": 0.4087,
                "qualitativeRisk": {"coding": [{"code": "moderate", "display": "Moderate risk"}]},
                "whenRange": {"low": {"value": 0.278}, "high": {"value": 0.551}},
                "rationale": "Phase 2B model output (fallback)",
            }],
            "note": [{"text": (
                "NeuroFusion-AD predicts 40.9% probability of amyloid positivity (MODERATE risk). "
                "RECOMMENDATION: Monitor per standard of care. "
                "NOTE: Investigational SaMD — not for clinical use."
            )}],
            "extension": [{"url": "...modality-importance", "extension": [
                {"url": "fluid", "valueDecimal": 0.143},
                {"url": "acoustic", "valueDecimal": 0.241},
                {"url": "motor", "valueDecimal": 0.218},
                {"url": "clinical", "valueDecimal": 0.398},
            ]}],
        },
    }
    return fallbacks[scenario_id]


def _generate_survival_curve(
    scenario_id: str,
    risk_assessment: dict,
) -> dict[str, Any]:
    """Generate synthetic Kaplan-Meier data based on predicted probability.

    In a real deployment this would use the Cox model output. For the demo
    we generate plausible survival curves based on the amyloid probability.
    """
    import math

    pred = risk_assessment.get("prediction", [{}])[0]
    prob = pred.get("probabilityDecimal", 0.5)

    # Progression-free survival probability at each month
    # High amyloid probability → faster progression to dementia
    hazard_scale = 0.008 + prob * 0.024  # baseline: 0.8-3.2% per month

    months = list(range(0, 61, 6))  # 0-60 months
    survival = []
    s = 1.0
    for t in months:
        if t > 0:
            s = s * math.exp(-hazard_scale * 6)
        survival.append(round(s, 4))

    # Comparison group: ADNI MCI median progression
    adni_hazard = 0.012
    adni_survival = []
    s2 = 1.0
    for t in months:
        if t > 0:
            s2 = s2 * math.exp(-adni_hazard * 6)
        adni_survival.append(round(s2, 4))

    return {
        "months": months,
        "patient_survival": survival,
        "adni_reference": adni_survival,
        "label": f"NeuroFusion-AD: p(amyloid)={prob:.2f}",
    }


# ---------------------------------------------------------------------------
# Serve frontend static files
# ---------------------------------------------------------------------------

# Mount static files (CSS, JS assets)
_assets_dir = FRONTEND_DIR / "dist"
if _assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(_assets_dir / "assets")), name="assets")


@app.get("/{full_path:path}", include_in_schema=False)
async def serve_frontend(full_path: str):
    """Serve the frontend SPA for all non-API routes."""
    # Try dist/ first (production build), then fall back to index.html in frontend/
    dist_index = FRONTEND_DIR / "dist" / "index.html"
    dev_index = FRONTEND_DIR / "index.html"

    if dist_index.exists():
        return FileResponse(str(dist_index))
    elif dev_index.exists():
        return FileResponse(str(dev_index))
    else:
        raise HTTPException(status_code=404, detail="Frontend not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "demo.backend.demo_api:app",
        host="0.0.0.0",
        port=int(os.environ.get("DEMO_PORT", "3000")),
        reload=False,
    )
