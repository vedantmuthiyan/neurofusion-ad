"""FHIR R4 RiskAssessment output builder for NeuroFusion-AD.

Converts raw model outputs (logits, probabilities, CI bounds, SHAP scores)
into a compliant FHIR R4 RiskAssessment resource.

FHIR RiskAssessment profile:
    resourceType: RiskAssessment
    status: final
    subject: Reference to input Patient
    prediction[0]: amyloid positivity probability (0-1)
    prediction[1]: low CI
    prediction[2]: high CI
    note: clinical recommendation text
    extension[]: modality importance scores (NeuroFusion-AD specific)

FHIR content type: application/fhir+json
SNOMED coding for amyloid status: 413448000 (Amyloid positivity)

IEC 62304 traceability: SRS-001 § 6.5, SAD-001 § 5.2
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# SNOMED codes for risk categories
_SNOMED_SYSTEM = "http://snomed.info/sct"
_LOINC_SYSTEM = "http://loinc.org"

# Risk category thresholds (Youden optimal from Phase 2B)
OPTIMAL_THRESHOLD = 0.6443  # from Phase 2B evaluation

_RISK_CATEGORIES = [
    (0.0,  0.35, "low",      "41931001",  "Low risk of amyloid positivity"),
    (0.35, 0.65, "moderate", "723509005", "Moderate risk of amyloid positivity"),
    (0.65, 1.01, "high",     "413448000", "High risk of amyloid positivity"),
]


class FHIROutputBuilder:
    """Builds FHIR R4 RiskAssessment resources from model output.

    Example:
        >>> builder = FHIROutputBuilder()
        >>> ra = builder.build(
        ...     patient_id="pt-123",
        ...     probability=0.78,
        ...     ci_lower=0.65,
        ...     ci_upper=0.89,
        ...     modality_importance={"fluid": 0.21, "clinical": 0.29},
        ... )
    """

    def build(
        self,
        patient_id: str,
        probability: float,
        ci_lower: float,
        ci_upper: float,
        modality_importance: dict[str, float],
        mmse_slope_pred: float | None = None,
        c_index: float | None = None,
        risk_assessment_id: str | None = None,
    ) -> dict[str, Any]:
        """Build FHIR RiskAssessment resource.

        Args:
            patient_id: Patient reference ID.
            probability: Amyloid positivity probability (0–1) after temperature scaling.
            ci_lower: 95% CI lower bound.
            ci_upper: 95% CI upper bound.
            modality_importance: Attention-weighted modality importance scores.
            mmse_slope_pred: Predicted MMSE slope (pts/year), optional.
            c_index: Model C-index for this patient's survival prediction, optional.
            risk_assessment_id: Optional ID for the resource (generated if not provided).

        Returns:
            FHIR R4 RiskAssessment resource dict.
        """
        import uuid
        ra_id = risk_assessment_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        risk_cat = _classify_risk(probability)

        prediction = self._build_prediction(
            probability=probability,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            risk_category=risk_cat,
        )

        note = _build_note(probability, risk_cat, mmse_slope_pred)

        resource: dict[str, Any] = {
            "resourceType": "RiskAssessment",
            "id": ra_id,
            "meta": {
                "versionId": "1",
                "lastUpdated": now,
                "profile": [
                    "https://neurofusion-ad.example.com/fhir/StructureDefinition/NeuroFusionRiskAssessment"
                ],
            },
            "status": "final",
            "method": {
                "coding": [{
                    "system": "https://neurofusion-ad.example.com/fhir/CodeSystem/method",
                    "code": "neurofusion-ad-v1",
                    "display": "NeuroFusion-AD Multimodal GNN (Phase 2B, embed_dim=256)",
                }]
            },
            "code": {
                "coding": [{
                    "system": _LOINC_SYSTEM,
                    "code": "75328-1",
                    "display": "Prognosis",
                }],
                "text": "Amyloid Positivity Risk Assessment",
            },
            "subject": {
                "reference": f"Patient/{patient_id}",
            },
            "occurrenceDateTime": now,
            "prediction": [prediction],
            "note": [{"text": note}],
        }

        # Add MMSE slope prediction if available
        if mmse_slope_pred is not None:
            resource["prediction"].append(
                _build_mmse_prediction(mmse_slope_pred)
            )

        # Add modality importance as extensions
        if modality_importance:
            resource["extension"] = _build_modality_extensions(
                modality_importance
            )

        # Add calibration metadata
        resource["extension"] = resource.get("extension", []) + [{
            "url": "https://neurofusion-ad.example.com/fhir/StructureDefinition/calibration",
            "extension": [
                {"url": "temperature", "valueDecimal": 0.756},
                {"url": "calibrationMethod", "valueString": "temperature-scaling"},
                {"url": "eceAfterCalibration", "valueDecimal": 0.0831},
            ]
        }]

        log.info(
            "FHIROutputBuilder.build complete",
            ra_id=ra_id,
            probability=round(probability, 4),
            risk_category=risk_cat["label"],
            n_predictions=len(resource["prediction"]),
        )
        return resource

    def _build_prediction(
        self,
        probability: float,
        ci_lower: float,
        ci_upper: float,
        risk_category: dict[str, str],
    ) -> dict[str, Any]:
        """Build the primary RiskAssessment.prediction entry.

        Args:
            probability: Point estimate probability.
            ci_lower: 95% CI lower bound.
            ci_upper: 95% CI upper bound.
            risk_category: Dict with 'label', 'snomed_code', 'display' keys.

        Returns:
            FHIR RiskAssessment.prediction dict.
        """
        return {
            "outcome": {
                "coding": [{
                    "system": _SNOMED_SYSTEM,
                    "code": risk_category["snomed_code"],
                    "display": risk_category["display"],
                }],
                "text": f"Amyloid positivity risk: {risk_category['label'].upper()}",
            },
            "probabilityDecimal": round(probability, 4),
            "qualitativeRisk": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/risk-probability",
                    "code": risk_category["fhir_risk_code"],
                    "display": risk_category["fhir_risk_display"],
                }]
            },
            "relativeRisk": round(probability / 0.44, 3),  # vs ADNI base rate 44%
            "whenRange": {
                "low": {
                    "value": round(ci_lower, 4),
                    "unit": "probability",
                    "system": "http://unitsofmeasure.org",
                    "code": "1",
                },
                "high": {
                    "value": round(ci_upper, 4),
                    "unit": "probability",
                    "system": "http://unitsofmeasure.org",
                    "code": "1",
                },
            },
            "rationale": (
                f"Multimodal GNN prediction (Phase 2B). "
                f"ADNI test AUC=0.8897, BH-001 test AUC=0.9071. "
                f"Temperature-scaled (T=0.756). "
                f"95% CI via Monte Carlo dropout."
            ),
        }


def _classify_risk(probability: float) -> dict[str, str]:
    """Classify amyloid risk into low/moderate/high categories.

    Thresholds informed by Youden's J optimal threshold (0.6443) from Phase 2B.

    Args:
        probability: Amyloid positivity probability.

    Returns:
        Dict with label, snomed_code, display, fhir_risk_code, fhir_risk_display.
    """
    for low, high, label, snomed, display in _RISK_CATEGORIES:
        if low <= probability < high:
            # Map to FHIR risk-probability codes
            fhir_map = {
                "low": ("negligible", "Negligible risk"),
                "moderate": ("moderate", "Moderate risk"),
                "high": ("high", "High risk"),
            }
            fhir_code, fhir_display = fhir_map[label]
            return {
                "label": label,
                "snomed_code": snomed,
                "display": display,
                "fhir_risk_code": fhir_code,
                "fhir_risk_display": fhir_display,
            }
    # Fallback
    return {
        "label": "high",
        "snomed_code": "413448000",
        "display": "High risk of amyloid positivity",
        "fhir_risk_code": "high",
        "fhir_risk_display": "High risk",
    }


def _build_note(probability: float, risk_cat: dict, mmse_slope: float | None) -> str:
    """Generate clinical recommendation note text.

    Args:
        probability: Amyloid positivity probability.
        risk_cat: Risk category dict.
        mmse_slope: Predicted MMSE slope (pts/yr), optional.

    Returns:
        Clinical note string.
    """
    pct = f"{probability * 100:.1f}%"
    label = risk_cat["label"].upper()

    note = (
        f"NeuroFusion-AD predicts {pct} probability of amyloid positivity ({label} risk). "
        f"Optimal classification threshold: {OPTIMAL_THRESHOLD:.2f} "
        f"(Youden's J, Phase 2B ADNI test: Sens=0.793, Spec=0.933). "
    )
    if mmse_slope is not None:
        note += f"Predicted MMSE change: {mmse_slope:+.2f} pts/year. "

    if probability >= OPTIMAL_THRESHOLD:
        note += (
            "RECOMMENDATION: Consider confirmatory amyloid PET or CSF testing. "
            "Clinical judgement and full history required before any clinical action."
        )
    else:
        note += (
            "RECOMMENDATION: Monitor per standard of care. "
            "This prediction does not exclude amyloid pathology."
        )

    note += (
        " NOTE: This is an investigational SaMD (FDA De Novo pathway, not yet cleared). "
        "Not for clinical use without authorized oversight."
    )
    return note


def _build_mmse_prediction(mmse_slope: float) -> dict[str, Any]:
    """Build FHIR prediction entry for MMSE slope.

    Args:
        mmse_slope: Predicted MMSE slope in pts/year.

    Returns:
        FHIR prediction dict.
    """
    return {
        "outcome": {
            "coding": [{
                "system": _LOINC_SYSTEM,
                "code": "72107-6",
                "display": "MMSE total score",
            }],
            "text": "Predicted MMSE trajectory",
        },
        "probabilityDecimal": None,
        "rationale": (
            f"Predicted MMSE slope: {mmse_slope:+.2f} pts/year "
            f"(Phase 2B RMSE=1.804 pts/year on ADNI test set N=66)"
        ),
    }


def _build_modality_extensions(
    modality_importance: dict[str, float],
) -> list[dict[str, Any]]:
    """Build FHIR extension array for modality importance scores.

    Args:
        modality_importance: Dict mapping modality names to importance floats.

    Returns:
        List of FHIR Extension resources.
    """
    ext_url = "https://neurofusion-ad.example.com/fhir/StructureDefinition/modality-importance"
    sub_exts = []
    for modality, score in modality_importance.items():
        sub_exts.append({
            "url": modality,
            "valueDecimal": round(float(score), 4),
        })
    return [{
        "url": ext_url,
        "extension": sub_exts,
    }]
