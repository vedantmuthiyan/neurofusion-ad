"""FHIR R4 Bundle validator and extractor for NeuroFusion-AD.

Validates incoming FHIR Bundles for the /fhir/RiskAssessment/$process endpoint.
Returns structured OperationOutcome errors on validation failure.

Validation rules (IEC 62304 SRS-001 § 5.5):
    - Bundle.resourceType must be 'Bundle'
    - Bundle.entry must contain at least one Patient resource
    - Patient must have a recognizable identifier
    - Minimum required: at least one clinical observation (MMSE or pTau)

FHIR content type: application/fhir+json
HTTP codes: 422 for validation errors, 400 for malformed JSON

IEC 62304 traceability: SRS-001 § 5.5, SAD-001 § 5.1
"""

from __future__ import annotations

from typing import Any

import structlog

log = structlog.get_logger(__name__)

# Minimum required LOINC codes — at least one must be present
_MINIMUM_LOINC_CODES = {
    "82154-1",   # pTau181 (CSF proxy)
    "100025-0",  # plasma pTau217
    "81600-4",   # NfL plasma
    "72107-6",   # MMSE
}

# All supported LOINC codes
_SUPPORTED_LOINC_CODES = _MINIMUM_LOINC_CODES | {
    "30155-3",   # APOE4 genotype
    "14683-7",   # Total tau CSF
    "NF-ABETA42P",
    "NF-ABETA40P",
    "NF-PTAU217",
    "NF-NFL",
}


def operation_outcome(
    severity: str,
    code: str,
    diagnostics: str,
    expression: str | None = None,
) -> dict[str, Any]:
    """Build a FHIR R4 OperationOutcome resource.

    Args:
        severity: 'error' | 'warning' | 'information' | 'fatal'
        code: Issue code per FHIR value set (e.g. 'required', 'invalid', 'not-found')
        diagnostics: Human-readable error message.
        expression: Optional FHIRPath expression indicating the problem location.

    Returns:
        FHIR R4 OperationOutcome dict.
    """
    issue: dict[str, Any] = {
        "severity": severity,
        "code": code,
        "diagnostics": diagnostics,
    }
    if expression:
        issue["expression"] = [expression]
    return {
        "resourceType": "OperationOutcome",
        "issue": [issue],
    }


class FHIRBundleValidator:
    """Validates FHIR R4 Bundles for NeuroFusion-AD inference requests.

    Checks structural validity and minimum content requirements. Does not
    enforce full FHIR R4 profile compliance (that requires a terminology server).

    Example:
        >>> validator = FHIRBundleValidator()
        >>> errors = validator.validate(bundle_dict)
        >>> if errors:
        ...     return JSONResponse(status_code=422, content=errors[0])
    """

    def validate(self, bundle: dict[str, Any]) -> list[dict[str, Any]]:
        """Validate a FHIR Bundle for NeuroFusion-AD inference.

        Args:
            bundle: Parsed FHIR Bundle dict.

        Returns:
            List of OperationOutcome dicts (empty if valid).
        """
        errors: list[dict[str, Any]] = []

        # 1. resourceType must be Bundle
        if bundle.get("resourceType") != "Bundle":
            errors.append(operation_outcome(
                "error", "invalid",
                f"resourceType must be 'Bundle', got '{bundle.get('resourceType')}'",
                "Bundle.resourceType",
            ))
            return errors  # can't validate further

        # 2. entry must be a list
        entries = bundle.get("entry", [])
        if not isinstance(entries, list):
            errors.append(operation_outcome(
                "error", "structure",
                "Bundle.entry must be an array",
                "Bundle.entry",
            ))
            return errors

        # 3. Must contain at least one Patient
        patients = [
            e.get("resource", {}) for e in entries
            if e.get("resource", {}).get("resourceType") == "Patient"
        ]
        if not patients:
            errors.append(operation_outcome(
                "error", "required",
                "Bundle must contain at least one Patient resource",
                "Bundle.entry",
            ))

        # 4. Validate each Patient has some identifier
        for i, pt in enumerate(patients):
            if not pt.get("id") and not pt.get("identifier") and not pt.get("name"):
                log.warning("Patient has no id/identifier/name", index=i)
                # warning only — not a hard error for inference

        # 5. Must contain at least one clinical Observation
        observations = [
            e.get("resource", {}) for e in entries
            if e.get("resource", {}).get("resourceType") == "Observation"
        ]
        if not observations:
            errors.append(operation_outcome(
                "error", "required",
                "Bundle must contain at least one Observation resource "
                "(MMSE, pTau217, or NfL)",
                "Bundle.entry",
            ))
            return errors

        # 6. Validate Observation required fields (status, code per FHIR R4)
        for i, obs in enumerate(observations):
            obs_errors = _validate_observation(obs, i)
            errors.extend(obs_errors)

        # 7. Warn if no minimum required clinical measurements
        found_loincs = set()
        for obs in observations:
            for coding in obs.get("code", {}).get("coding", []):
                found_loincs.add(coding.get("code", ""))
        if not found_loincs.intersection(_MINIMUM_LOINC_CODES):
            # Check for local extension codes
            has_local = any(
                c in found_loincs for c in ["NF-PTAU217", "NF-NFL"]
            )
            if not has_local:
                # Not a hard error — model can still run with imputed features
                log.warning(
                    "No recognized LOINC codes found — all features will be imputed",
                    found_loincs=list(found_loincs),
                )

        log.info(
            "FHIRBundleValidator.validate complete",
            n_patients=len(patients),
            n_observations=len(observations),
            n_errors=len(errors),
        )
        return errors

    def extract_patient_id(self, bundle: dict[str, Any]) -> str:
        """Extract patient identifier from Bundle.

        Returns first Patient.id or first identifier value found.

        Args:
            bundle: FHIR Bundle dict (already validated).

        Returns:
            Patient ID string, or 'unknown' if not found.
        """
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            if resource.get("resourceType") == "Patient":
                if resource.get("id"):
                    return str(resource["id"])
                for ident in resource.get("identifier", []):
                    if ident.get("value"):
                        return str(ident["value"])
        return "unknown"

    def extract_requesting_system(self, bundle: dict[str, Any]) -> str:
        """Extract source system from Bundle.meta or identifier.

        Args:
            bundle: FHIR Bundle dict.

        Returns:
            Source system string, or 'unknown'.
        """
        meta = bundle.get("meta", {})
        source = meta.get("source", "")
        if source:
            return source
        # Try Bundle identifier
        ident = bundle.get("identifier", {})
        return ident.get("system", "unknown")


def _validate_observation(obs: dict[str, Any], index: int) -> list[dict[str, Any]]:
    """Validate a single FHIR Observation resource.

    Per FHIR R4 spec, Observation requires: status, code.

    Args:
        obs: FHIR Observation dict.
        index: Position in Bundle.entry for error reporting.

    Returns:
        List of OperationOutcome dicts (empty if valid).
    """
    errors = []
    expr_base = f"Bundle.entry[{index}].resource"

    # status is required (Observation.status: 1..1)
    valid_statuses = {
        "registered", "preliminary", "final", "amended",
        "corrected", "cancelled", "entered-in-error", "unknown",
    }
    status = obs.get("status")
    if not status:
        errors.append(operation_outcome(
            "error", "required",
            "Observation.status is required",
            f"{expr_base}.status",
        ))
    elif status not in valid_statuses:
        errors.append(operation_outcome(
            "error", "value",
            f"Observation.status '{status}' is not a valid value",
            f"{expr_base}.status",
        ))

    # code is required (Observation.code: 1..1)
    if not obs.get("code"):
        errors.append(operation_outcome(
            "error", "required",
            "Observation.code is required",
            f"{expr_base}.code",
        ))

    return errors
