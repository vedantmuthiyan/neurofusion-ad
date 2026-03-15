# NeuroFusion-AD Clinical Demo

Interactive demonstration of the NeuroFusion-AD multimodal GNN for Alzheimer's
Disease amyloid positivity risk assessment.

## Scenarios

| Patient | Age | Sex | MMSE | pTau217 | NfL | APOE | Expected Risk |
|---------|-----|-----|------|---------|-----|------|---------------|
| Margaret Chen | 77 | F | 22 | 0.85 pg/mL | 18.5 pg/mL | ε3/ε4 | HIGH (≈94%) |
| Robert Martinez | 65 | M | 26 | 0.45 pg/mL | 12.1 pg/mL | ε3/ε3 | MODERATE (≈52%) |
| Dorothy Walsh | 82 | F | 24 | 0.32 pg/mL | 15.0 pg/mL | ε4/ε4 | MODERATE (≈41%) |

## Quick Start

### Option 1 — Run locally (requires main API at localhost:8000)

```bash
# From project root:
pip install fastapi uvicorn httpx structlog python-multipart
python demo/backend/demo_api.py
# → http://localhost:3000
```

### Option 2 — Docker (self-contained)

```bash
# Start main API stack first:
docker-compose up -d

# Then start demo:
docker-compose -f demo/docker-compose.demo.yml up --build -d
# → http://localhost:3000
```

## Components

- **RiskGauge** — SVG half-circle gauge showing amyloid probability + 95% CI band
- **ModalityImportanceChart** — Horizontal bar chart with attention-weighted modality contributions
- **KaplanMeierCurve** — Progression-free survival curve vs. ADNI reference
- **AlertBanner** — Color-coded recommendation banner (green/orange/red)

## Architecture

```
Browser → Demo Backend (port 3000)
              ↓ POST /fhir/RiskAssessment/$process
         FHIR API (port 8000) → NeuroFusion-AD model
```

The demo backend also serves the frontend SPA and pre-computed fallback results
if the FHIR API is unavailable.

## Disclaimer

**INVESTIGATIONAL DEVICE** — FDA De Novo pathway (pending). This output is for
research/demonstration purposes only. Not cleared for clinical decision-making.
