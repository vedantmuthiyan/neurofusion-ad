"""Integration tests for NeuroFusion-AD FHIR API.

Tests are designed to run WITHOUT a loaded model checkpoint:
  - /health and /ready endpoints
  - FHIR Bundle validation (400/422 error paths)
  - FHIR output builder (unit-level, no model needed)
  - Temperature scaling unit tests

Uses httpx.AsyncClient with ASGITransport (compatible with httpx>=0.28).

To run against a live stack with a loaded model:
    MODEL_PATH=... SCALER_PATH=... pytest tests/integration/ -v -m live

IEC 62304 traceability: SRS-001 § 5.5, SAD-001 § 5.1
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
import torch

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

MINIMAL_BUNDLE: dict[str, Any] = {
    "resourceType": "Bundle",
    "type": "collection",
    "entry": [
        {
            "resource": {
                "resourceType": "Patient",
                "id": "pt-test-001",
                "identifier": [{"system": "urn:test", "value": "pt-test-001"}],
                "gender": "female",
                "birthDate": "1948-05-12",
            }
        },
        {
            "resource": {
                "resourceType": "Observation",
                "status": "final",
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "72107-6",
                            "display": "MMSE total score",
                        }
                    ]
                },
                "subject": {"reference": "Patient/pt-test-001"},
                "valueQuantity": {"value": 22.0, "unit": "score"},
            }
        },
        {
            "resource": {
                "resourceType": "Observation",
                "status": "final",
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "100025-0",
                            "display": "Plasma pTau217",
                        }
                    ]
                },
                "subject": {"reference": "Patient/pt-test-001"},
                "valueQuantity": {"value": 0.85, "unit": "pg/mL"},
            }
        },
        {
            "resource": {
                "resourceType": "Observation",
                "status": "final",
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "81600-4",
                            "display": "NfL plasma",
                        }
                    ]
                },
                "subject": {"reference": "Patient/pt-test-001"},
                "valueQuantity": {"value": 18.5, "unit": "pg/mL"},
            }
        },
    ],
}


# ---------------------------------------------------------------------------
# App fixture — patches model loading so no checkpoint needed
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def patched_app():
    """Return the FastAPI app with model loading patched out."""
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = mock_model
    mock_model.train.return_value = mock_model
    mock_model.return_value = {
        "amyloid_logit": torch.tensor(0.85),
        "mmse_slope": torch.tensor(-1.2),
        "cox_log_hazard": torch.tensor(0.3),
    }

    mock_prep = MagicMock()
    mock_prep.from_fhir_bundle.return_value = {
        "fluid": torch.zeros(1, 2),
        "acoustic": torch.zeros(1, 12),
        "motor": torch.zeros(1, 8),
        "clinical": torch.zeros(1, 10),
    }

    with (
        patch("src.api.main.torch.load", return_value={
            "config": {"embed_dim": 256, "num_heads": 4, "dropout": 0.4},
            "model_state_dict": {},
        }),
        patch("src.api.main.NeuroFusionAD", return_value=mock_model),
        patch("src.api.main.InferencePreprocessor.from_scaler", return_value=mock_prep),
    ):
        from src.api.main import app
        import src.api.main as main_module

        main_module._model_ready = True
        main_module._model = mock_model
        main_module._preprocessor = mock_prep

        yield app, main_module


@pytest.fixture
async def client(patched_app):
    """Async HTTPX client using ASGITransport — no server needed."""
    app, _ = patched_app
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as c:
        yield c, patched_app[1]


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@pytest.mark.anyio
class TestHealth:
    async def test_health_returns_200(self, client):
        c, _ = client
        resp = await c.get("/health")
        assert resp.status_code == 200

    async def test_health_body_fields(self, client):
        c, _ = client
        body = (await c.get("/health")).json()
        assert body["status"] == "ok"
        assert body["service"] == "neurofusion-ad-api"
        assert "version" in body
        assert "timestamp" in body

    async def test_health_content_type(self, client):
        c, _ = client
        resp = await c.get("/health")
        assert "application/json" in resp.headers["content-type"]


# ---------------------------------------------------------------------------
# /ready
# ---------------------------------------------------------------------------

@pytest.mark.anyio
class TestReady:
    async def test_ready_when_model_loaded(self, client):
        c, m = client
        m._model_ready = True
        resp = await c.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"

    async def test_ready_503_when_not_loaded(self, client):
        c, m = client
        original = m._model_ready
        m._model_ready = False
        try:
            resp = await c.get("/ready")
            assert resp.status_code == 503
        finally:
            m._model_ready = original


# ---------------------------------------------------------------------------
# /fhir/RiskAssessment/$process — validation error paths
# ---------------------------------------------------------------------------

ENDPOINT = "/fhir/RiskAssessment/$process"


@pytest.mark.anyio
class TestFHIRValidation:
    async def test_malformed_json_returns_400(self, client):
        c, _ = client
        resp = await c.post(
            ENDPOINT,
            content=b"not json {{{",
            headers={"Content-Type": "application/fhir+json"},
        )
        assert resp.status_code == 400
        assert resp.json()["resourceType"] == "OperationOutcome"

    async def test_wrong_resource_type_returns_422(self, client):
        c, _ = client
        resp = await c.post(ENDPOINT, json={"resourceType": "Patient", "id": "x"})
        assert resp.status_code == 422
        body = resp.json()
        assert body["resourceType"] == "OperationOutcome"
        assert any("Bundle" in issue["diagnostics"] for issue in body["issue"])

    async def test_missing_patient_returns_422(self, client):
        c, _ = client
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Observation",
                        "status": "final",
                        "code": {"coding": [{"system": "http://loinc.org", "code": "72107-6"}]},
                    }
                }
            ],
        }
        resp = await c.post(ENDPOINT, json=bundle)
        assert resp.status_code == 422

    async def test_missing_observation_returns_422(self, client):
        c, _ = client
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [{"resource": {"resourceType": "Patient", "id": "pt-x"}}],
        }
        resp = await c.post(ENDPOINT, json=bundle)
        assert resp.status_code == 422

    async def test_observation_missing_status_returns_422(self, client):
        c, _ = client
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {"resource": {"resourceType": "Patient", "id": "pt-x"}},
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"coding": [{"system": "http://loinc.org", "code": "72107-6"}]},
                    }
                },
            ],
        }
        resp = await c.post(ENDPOINT, json=bundle)
        assert resp.status_code == 422
        body = resp.json()
        assert any("status" in issue["diagnostics"] for issue in body["issue"])

    async def test_observation_invalid_status_returns_422(self, client):
        c, _ = client
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {"resource": {"resourceType": "Patient", "id": "pt-x"}},
                {
                    "resource": {
                        "resourceType": "Observation",
                        "status": "not-a-valid-status",
                        "code": {"coding": [{"system": "http://loinc.org", "code": "72107-6"}]},
                    }
                },
            ],
        }
        resp = await c.post(ENDPOINT, json=bundle)
        assert resp.status_code == 422

    async def test_observation_missing_code_returns_422(self, client):
        c, _ = client
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {"resource": {"resourceType": "Patient", "id": "pt-x"}},
                {"resource": {"resourceType": "Observation", "status": "final"}},
            ],
        }
        resp = await c.post(ENDPOINT, json=bundle)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /fhir/RiskAssessment/$process — happy path (mocked inference)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
class TestFHIRInference:
    async def test_valid_bundle_returns_200(self, client):
        c, _ = client
        resp = await c.post(ENDPOINT, json=MINIMAL_BUNDLE)
        assert resp.status_code == 200

    async def test_response_is_risk_assessment(self, client):
        c, _ = client
        body = (await c.post(ENDPOINT, json=MINIMAL_BUNDLE)).json()
        assert body["resourceType"] == "RiskAssessment"
        assert body["status"] == "final"

    async def test_response_has_prediction(self, client):
        c, _ = client
        body = (await c.post(ENDPOINT, json=MINIMAL_BUNDLE)).json()
        assert "prediction" in body
        assert len(body["prediction"]) >= 1
        pred = body["prediction"][0]
        assert "probabilityDecimal" in pred
        assert 0.0 <= pred["probabilityDecimal"] <= 1.0

    async def test_response_has_ci_bounds(self, client):
        c, _ = client
        body = (await c.post(ENDPOINT, json=MINIMAL_BUNDLE)).json()
        ci = body["prediction"][0]["whenRange"]
        assert ci["low"]["value"] <= ci["high"]["value"]

    async def test_response_has_patient_subject(self, client):
        c, _ = client
        body = (await c.post(ENDPOINT, json=MINIMAL_BUNDLE)).json()
        assert body["subject"]["reference"] == "Patient/pt-test-001"

    async def test_response_has_note(self, client):
        c, _ = client
        body = (await c.post(ENDPOINT, json=MINIMAL_BUNDLE)).json()
        assert "note" in body and len(body["note"]) >= 1
        assert "text" in body["note"][0]

    async def test_response_has_modality_extensions(self, client):
        c, _ = client
        body = (await c.post(ENDPOINT, json=MINIMAL_BUNDLE)).json()
        ext_urls = [e.get("url", "") for e in body.get("extension", [])]
        assert any("modality-importance" in u for u in ext_urls)

    async def test_response_content_type_is_fhir_json(self, client):
        c, _ = client
        resp = await c.post(ENDPOINT, json=MINIMAL_BUNDLE)
        assert "application/fhir+json" in resp.headers["content-type"]

    async def test_response_has_request_id_header(self, client):
        c, _ = client
        resp = await c.post(
            ENDPOINT,
            json=MINIMAL_BUNDLE,
            headers={"X-Request-ID": "test-req-id-123"},
        )
        assert resp.headers.get("x-request-id") == "test-req-id-123"

    async def test_response_has_latency_header(self, client):
        c, _ = client
        resp = await c.post(ENDPOINT, json=MINIMAL_BUNDLE)
        assert "x-latency-ms" in resp.headers
        assert float(resp.headers["x-latency-ms"]) >= 0

    async def test_response_has_etag_header(self, client):
        c, _ = client
        resp = await c.post(ENDPOINT, json=MINIMAL_BUNDLE)
        assert "etag" in resp.headers

    async def test_api_key_auth_rejected_when_configured(self, client):
        c, m = client
        original = m.API_KEY
        m.API_KEY = "secret-test-key"
        try:
            resp = await c.post(
                ENDPOINT,
                json=MINIMAL_BUNDLE,
                headers={"X-API-Key": "wrong-key"},
            )
            assert resp.status_code == 401
            assert resp.json()["resourceType"] == "OperationOutcome"
        finally:
            m.API_KEY = original

    async def test_api_key_auth_accepted_with_correct_key(self, client):
        c, m = client
        original = m.API_KEY
        m.API_KEY = "secret-test-key"
        try:
            resp = await c.post(
                ENDPOINT,
                json=MINIMAL_BUNDLE,
                headers={"X-API-Key": "secret-test-key"},
            )
            assert resp.status_code == 200
        finally:
            m.API_KEY = original

    async def test_model_not_ready_returns_503(self, client):
        c, m = client
        original = m._model_ready
        m._model_ready = False
        try:
            resp = await c.post(ENDPOINT, json=MINIMAL_BUNDLE)
            assert resp.status_code == 503
        finally:
            m._model_ready = original


# ---------------------------------------------------------------------------
# FHIR output builder — synchronous unit tests (no model, no API)
# ---------------------------------------------------------------------------

class TestFHIROutputBuilder:
    def setup_method(self):
        from src.api.fhir_output import FHIROutputBuilder
        self.builder = FHIROutputBuilder()

    def test_build_returns_risk_assessment(self):
        ra = self.builder.build(
            patient_id="pt-123",
            probability=0.78,
            ci_lower=0.65,
            ci_upper=0.89,
            modality_importance={"fluid": 0.21, "clinical": 0.29},
        )
        assert ra["resourceType"] == "RiskAssessment"

    def test_high_probability_classified_high(self):
        ra = self.builder.build(
            patient_id="pt-123",
            probability=0.80,
            ci_lower=0.70,
            ci_upper=0.90,
            modality_importance={},
        )
        assert "HIGH" in ra["prediction"][0]["outcome"]["text"]

    def test_low_probability_classified_low(self):
        ra = self.builder.build(
            patient_id="pt-123",
            probability=0.20,
            ci_lower=0.10,
            ci_upper=0.30,
            modality_importance={},
        )
        assert "LOW" in ra["prediction"][0]["outcome"]["text"]

    def test_moderate_probability_classified_moderate(self):
        ra = self.builder.build(
            patient_id="pt-123",
            probability=0.50,
            ci_lower=0.40,
            ci_upper=0.60,
            modality_importance={},
        )
        assert "MODERATE" in ra["prediction"][0]["outcome"]["text"]

    def test_mmse_slope_adds_second_prediction(self):
        ra = self.builder.build(
            patient_id="pt-123",
            probability=0.75,
            ci_lower=0.60,
            ci_upper=0.88,
            modality_importance={},
            mmse_slope_pred=-1.5,
        )
        assert len(ra["prediction"]) == 2

    def test_ci_bounds_in_range(self):
        ra = self.builder.build(
            patient_id="pt-x",
            probability=0.65,
            ci_lower=0.55,
            ci_upper=0.75,
            modality_importance={},
        )
        low = ra["prediction"][0]["whenRange"]["low"]["value"]
        high = ra["prediction"][0]["whenRange"]["high"]["value"]
        assert 0.0 <= low <= 1.0
        assert 0.0 <= high <= 1.0
        assert low <= high

    def test_calibration_extension_present(self):
        ra = self.builder.build(
            patient_id="pt-x",
            probability=0.65,
            ci_lower=0.55,
            ci_upper=0.75,
            modality_importance={},
        )
        calib = [e for e in ra.get("extension", []) if "calibration" in e.get("url", "")]
        assert len(calib) == 1
        sub = {s["url"]: s for s in calib[0]["extension"]}
        assert sub["temperature"]["valueDecimal"] == pytest.approx(0.756, abs=1e-3)

    def test_note_contains_investigational_disclaimer(self):
        ra = self.builder.build(
            patient_id="pt-x",
            probability=0.65,
            ci_lower=0.55,
            ci_upper=0.75,
            modality_importance={},
        )
        note = ra["note"][0]["text"].lower()
        assert "investigational" in note or "not yet cleared" in note


# ---------------------------------------------------------------------------
# FHIR validator — synchronous unit tests
# ---------------------------------------------------------------------------

class TestFHIRValidator:
    def setup_method(self):
        from src.api.fhir_validator import FHIRBundleValidator
        self.v = FHIRBundleValidator()

    def test_valid_bundle_no_errors(self):
        assert self.v.validate(MINIMAL_BUNDLE) == []

    def test_wrong_resource_type_returns_error(self):
        errors = self.v.validate({"resourceType": "Patient"})
        assert len(errors) == 1
        assert errors[0]["issue"][0]["code"] == "invalid"

    def test_no_patient_returns_error(self):
        bundle = {
            "resourceType": "Bundle",
            "entry": [
                {"resource": {
                    "resourceType": "Observation",
                    "status": "final",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "72107-6"}]},
                }}
            ],
        }
        errors = self.v.validate(bundle)
        assert any("Patient" in e["issue"][0]["diagnostics"] for e in errors)

    def test_no_observation_returns_error(self):
        bundle = {
            "resourceType": "Bundle",
            "entry": [{"resource": {"resourceType": "Patient", "id": "x"}}],
        }
        errors = self.v.validate(bundle)
        assert any("Observation" in e["issue"][0]["diagnostics"] for e in errors)

    def test_extract_patient_id(self):
        assert self.v.extract_patient_id(MINIMAL_BUNDLE) == "pt-test-001"

    def test_extract_patient_id_unknown_when_empty(self):
        pid = self.v.extract_patient_id({"resourceType": "Bundle", "entry": []})
        assert pid == "unknown"

    def test_extract_requesting_system(self):
        bundle = {
            "resourceType": "Bundle",
            "meta": {"source": "https://ehr.example.com"},
            "entry": [],
        }
        assert self.v.extract_requesting_system(bundle) == "https://ehr.example.com"


# ---------------------------------------------------------------------------
# Temperature scaling — synchronous unit tests
# ---------------------------------------------------------------------------

class TestTemperatureScaling:
    def test_positive_logit_gives_prob_above_05(self):
        from src.api.main import _temperature_scale
        assert _temperature_scale(1.0, temperature=1.0) > 0.5

    def test_zero_logit_gives_05(self):
        from src.api.main import _temperature_scale
        assert _temperature_scale(0.0, temperature=1.0) == pytest.approx(0.5, abs=1e-6)

    def test_higher_temperature_flattens_probability(self):
        from src.api.main import _temperature_scale
        assert _temperature_scale(2.0, 0.5) > _temperature_scale(2.0, 2.0)

    def test_phase2b_temperature(self):
        """T=0.756 should push a logit of 0.5 above 0.60."""
        from src.api.main import _temperature_scale
        prob = _temperature_scale(0.5, temperature=0.756)
        assert 0.60 < prob < 0.70
