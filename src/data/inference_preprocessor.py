"""Single-patient inference preprocessor for NeuroFusion-AD FHIR API.

Converts raw clinical values (from FHIR Bundle or structured dict) into the
normalized tensor batch expected by NeuroFusionAD.forward().

Feature schema (Phase 2B — matches training data):
    fluid [2]:    ptau217 (pg/mL), nfl_plasma (pg/mL) — z-score normalized
    acoustic [12]: 12 acoustic/speech features — z-score normalized (or imputed)
    motor [8]:    8 motor/digital features — z-score normalized (or imputed)
    clinical [10]: age, sex_code, edu_years, mmse_baseline, apoe4_count, tau_csf,
                   abeta42_plasma, abeta40_plasma, ptau181_tau_ratio,
                   abeta4240_plasma_ratio

Normalization uses the training scaler (data/processed/adni/scaler.pkl).
Missing features are imputed with per-column training medians.

FHIR LOINC mappings:
    pTau181 (ADNI proxy): 82154-1
    plasma pTau217: 100025-0 (pending) | use local extension
    NfL plasma: 81600-4
    MMSE total score: 72107-6
    APOE genotype: 30155-3
    Total tau CSF: 14683-7
    Abeta42 plasma: local (no standard LOINC)
    Abeta40 plasma: local (no standard LOINC)

IEC 62304 traceability: SRS-001 § 5.4, SAD-001 § 5.0
"""

from __future__ import annotations

import hashlib
import pickle
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# FHIR LOINC code → internal feature mapping
# ---------------------------------------------------------------------------

LOINC_TO_FEATURE: dict[str, str] = {
    "82154-1": "PTAU217",       # pTau181 CSF (ADNI proxy for pTau217)
    "100025-0": "PTAU217",      # plasma pTau217 (Bio-Hermes, pending LOINC)
    "81600-4": "NFL_PLASMA",    # NfL plasma
    "72107-6": "MMSE_BASELINE", # MMSE total score
    "30155-3": "APOE4_COUNT",   # APOE4 allele count
    "14683-7": "TAU_CSF",       # Total tau CSF
    # Local extension codes (EHR-specific, no standard LOINC)
    "NF-ABETA42P": "ABETA42_PLASMA",
    "NF-ABETA40P": "ABETA40_PLASMA",
    "NF-PTAU217": "PTAU217",
    "NF-NFL": "NFL_PLASMA",
}

# APOE4 genotype string → allele count
APOE4_GENOTYPE_TO_COUNT: dict[str, int] = {
    "e2/e2": 0, "e2/e3": 0, "e3/e3": 0,
    "e2/e4": 1, "e3/e4": 1,
    "e4/e4": 2,
    # Numeric fallbacks
    "0": 0, "1": 1, "2": 2,
}

# Default imputation values (median from ADNI training set, approximate)
_IMPUTE_DEFAULTS: dict[str, float] = {
    "PTAU217": 0.0,         # z-score normalized median ≈ 0
    "NFL_PLASMA": 0.0,
    "AGE": 0.0,             # z-score (mean=73.2, std=7.6 in ADNI)
    "SEX_CODE": 0.5,        # roughly even split
    "EDUCATION_YEARS": 0.0,
    "MMSE_BASELINE": 0.0,
    "APOE4_COUNT": 0.0,
    "TAU_CSF": 0.0,
    "ABETA42_PLASMA": 0.0,
    "ABETA40_PLASMA": 0.0,
}


class InferencePreprocessor:
    """Converts raw patient data into NeuroFusionAD input tensors.

    Attributes:
        scaler_path: Path to scaler.pkl fitted on ADNI training set.
        imputation_stats: Per-column training medians for missing feature imputation.

    Example:
        >>> prep = InferencePreprocessor.from_scaler("data/processed/adni/scaler.pkl")
        >>> batch = prep.from_dict({
        ...     "age": 72.0, "sex": "female", "ptau217": 4.5, "nfl": 12.0,
        ...     "mmse": 26, "apoe4_count": 1,
        ... })
        >>> model_outputs = model(batch)
    """

    def __init__(
        self,
        scaler: Any,
        imputation_stats: dict[str, float] | None = None,
    ) -> None:
        """Initialise with a fitted StandardScaler.

        Args:
            scaler: Fitted sklearn StandardScaler from training.
            imputation_stats: Per-column median values; fallback to 0 if None.
        """
        self.scaler = scaler
        self.feature_names: list[str] = list(
            getattr(scaler, "feature_names_in_", [])
        )
        self.imputation_stats = imputation_stats or {}
        log.info(
            "InferencePreprocessor initialised",
            n_scaler_features=len(self.feature_names),
        )

    @classmethod
    def from_scaler(cls, scaler_path: str) -> "InferencePreprocessor":
        """Load from a saved scaler.pkl file.

        Args:
            scaler_path: Path to scaler.pkl.

        Returns:
            InferencePreprocessor instance.

        Raises:
            FileNotFoundError: If scaler_path does not exist.
        """
        path = Path(scaler_path)
        if not path.exists():
            raise FileNotFoundError(f"Scaler not found: {path}")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(path, "rb") as f:
                scaler = pickle.load(f)
        return cls(scaler)

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def _normalize(self, col: str, raw_value: float) -> float:
        """Z-score normalize a single feature value using the training scaler.

        Args:
            col: Column name as stored in scaler.feature_names_in_.
            raw_value: Raw (un-normalized) clinical measurement.

        Returns:
            Z-score normalized value. Returns 0.0 if column not in scaler.
        """
        if col not in self.feature_names:
            return 0.0
        idx = self.feature_names.index(col)
        mean = float(self.scaler.mean_[idx])
        std = float(self.scaler.scale_[idx])
        if std < 1e-8:
            return 0.0
        return (raw_value - mean) / std

    def _get_normalized(self, col: str, raw_values: dict[str, float]) -> float:
        """Get normalized value for a column from raw input dict.

        Falls back to imputation stats then default (0.0).

        Args:
            col: Feature column name.
            raw_values: Dict of raw clinical values.

        Returns:
            Normalized float value.
        """
        if col in raw_values and not np.isnan(raw_values[col]):
            return self._normalize(col, raw_values[col])
        # Fallback: use imputation median (already normalized in training)
        return self.imputation_stats.get(col, _IMPUTE_DEFAULTS.get(col, 0.0))

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def from_dict(
        self,
        patient: dict[str, Any],
        patient_id: str = "unknown",
    ) -> dict[str, torch.Tensor]:
        """Build model input batch from a structured patient dict.

        Accepts both raw clinical values (which get normalized) and
        pre-normalized values (passed through directly).

        Args:
            patient: Dict with any subset of keys:
                age (years), sex ('male'|'female'|0|1),
                edu_years, mmse, apoe4_count (0|1|2),
                ptau217 (pg/mL), nfl (pg/mL),
                tau_csf (pg/mL), ptau181_csf (pg/mL),
                abeta42_plasma (pg/mL), abeta40_plasma (pg/mL),
                acoustic [12 floats], motor [8 floats].
            patient_id: Opaque patient identifier (will be SHA-256 hashed).

        Returns:
            Batch dict compatible with NeuroFusionAD.forward():
                {fluid: [1,2], acoustic: [1,12], motor: [1,8], clinical: [1,10]}
        """
        raw: dict[str, float] = {}

        # -- Demographics -----------------------------------------------
        age = float(patient.get("age", float("nan")))
        if not np.isnan(age):
            raw["AGE"] = age

        sex_raw = patient.get("sex", patient.get("gender", 1))
        if isinstance(sex_raw, str):
            sex_code = 0.0 if sex_raw.lower() in ("f", "female") else 1.0
        else:
            sex_code = float(sex_raw)

        edu = float(patient.get("edu_years", patient.get("education_years", float("nan"))))
        if not np.isnan(edu):
            raw["EDUCATION_YEARS"] = edu

        mmse = float(patient.get("mmse", patient.get("mmse_baseline", float("nan"))))
        if not np.isnan(mmse):
            raw["MMSE_BASELINE"] = mmse

        apoe4 = float(patient.get("apoe4_count", float("nan")))
        if not np.isnan(apoe4):
            raw["APOE4_COUNT"] = apoe4

        # -- CSF / plasma biomarkers ------------------------------------
        ptau217 = float(patient.get("ptau217", float("nan")))
        if not np.isnan(ptau217):
            raw["PTAU217"] = ptau217

        nfl = float(patient.get("nfl", patient.get("nfl_plasma", float("nan"))))
        if not np.isnan(nfl):
            raw["NFL_PLASMA"] = nfl

        tau_csf = float(patient.get("tau_csf", float("nan")))
        if not np.isnan(tau_csf):
            raw["TAU_CSF"] = tau_csf

        abeta42p = float(patient.get("abeta42_plasma", float("nan")))
        if not np.isnan(abeta42p):
            raw["ABETA42_PLASMA"] = abeta42p

        abeta40p = float(patient.get("abeta40_plasma", float("nan")))
        if not np.isnan(abeta40p):
            raw["ABETA40_PLASMA"] = abeta40p

        # -- Build normalized feature vectors ---------------------------
        fluid = [
            self._get_normalized("PTAU217", raw),
            self._get_normalized("NFL_PLASMA", raw),
        ]

        # Acoustic (12): use provided array or zeros (synthetic in ADNI)
        acoustic_raw = patient.get("acoustic", None)
        if acoustic_raw is not None and len(acoustic_raw) == 12:
            acoustic = [float(v) for v in acoustic_raw]
        else:
            acoustic = [0.0] * 12

        # Motor (8): use provided array or zeros (synthetic in ADNI)
        motor_raw = patient.get("motor", None)
        if motor_raw is not None and len(motor_raw) == 8:
            motor = [float(v) for v in motor_raw]
        else:
            motor = [0.0] * 8

        # Clinical (10): 8 base + 2 derived
        age_norm = self._get_normalized("AGE", raw)
        edu_norm = self._get_normalized("EDUCATION_YEARS", raw)
        mmse_norm = self._get_normalized("MMSE_BASELINE", raw)
        apoe4_norm = self._get_normalized("APOE4_COUNT", raw) if "APOE4_COUNT" in raw else float(apoe4 if not np.isnan(apoe4) else 0.0)
        tau_csf_norm = self._get_normalized("TAU_CSF", raw)
        abeta42p_norm = self._get_normalized("ABETA42_PLASMA", raw)
        abeta40p_norm = self._get_normalized("ABETA40_PLASMA", raw)

        # Derived feature: pTau181/totalTau ratio
        ptau181_csf = float(patient.get("ptau181_csf", 0.0))
        tau_csf_val = float(patient.get("tau_csf", 1.0))
        ptau_tau_ratio = ptau181_csf / (tau_csf_val + 1e-6) if tau_csf_val > 0 else 0.0

        # Derived feature: plasma Abeta40/42 ratio
        abeta42p_val = float(patient.get("abeta42_plasma", 1.0))
        abeta40p_val = float(patient.get("abeta40_plasma", 0.0))
        abeta4240_ratio = abeta40p_val / (abeta42p_val + 1e-6) if abeta42p_val > 0 else 0.0

        clinical = [
            age_norm, sex_code, edu_norm, mmse_norm,
            apoe4_norm, tau_csf_norm, abeta42p_norm, abeta40p_norm,
            ptau_tau_ratio, abeta4240_ratio,
        ]

        # -- Convert to tensors -----------------------------------------
        batch: dict[str, torch.Tensor] = {
            "fluid": torch.tensor([fluid], dtype=torch.float32),
            "acoustic": torch.tensor([acoustic], dtype=torch.float32),
            "motor": torch.tensor([motor], dtype=torch.float32),
            "clinical": torch.tensor([clinical], dtype=torch.float32),
        }

        log.debug(
            "InferencePreprocessor.from_dict complete",
            patient_id=_hash_id(patient_id),
            fluid_norm=round(fluid[0], 3),
            mmse_raw=mmse,
            age_raw=age,
        )
        return batch

    def from_fhir_bundle(
        self,
        bundle: dict[str, Any],
        patient_id: str = "unknown",
    ) -> dict[str, torch.Tensor]:
        """Build model input batch from a FHIR R4 Bundle.

        Extracts Observation resources by LOINC code and Patient demographics.
        Missing observations are imputed with training medians.

        Args:
            bundle: FHIR R4 Bundle dict with 'resourceType': 'Bundle' and 'entry' list.
            patient_id: Opaque patient ID for audit logging.

        Returns:
            Batch dict compatible with NeuroFusionAD.forward().

        Raises:
            ValueError: If bundle is not a FHIR Bundle resourceType.
        """
        if bundle.get("resourceType") != "Bundle":
            raise ValueError(
                f"Expected FHIR Bundle, got resourceType='{bundle.get('resourceType')}'"
            )

        patient_data: dict[str, Any] = {}

        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            rtype = resource.get("resourceType")

            if rtype == "Patient":
                # Extract age from birthDate
                birth_date = resource.get("birthDate")
                if birth_date:
                    try:
                        bd = date.fromisoformat(str(birth_date))
                        today = date.today()
                        age_years = (today - bd).days / 365.25
                        patient_data["age"] = age_years
                    except (ValueError, TypeError):
                        pass
                # Extract sex
                gender = resource.get("gender", "")
                patient_data["sex"] = gender

            elif rtype == "Observation":
                loinc_code = _extract_loinc_code(resource)
                if loinc_code is None:
                    continue
                feat_name = LOINC_TO_FEATURE.get(loinc_code)
                if feat_name is None:
                    continue
                value = _extract_observation_value(resource)
                if value is not None:
                    # Map internal feature names to patient_data keys
                    key_map = {
                        "PTAU217": "ptau217",
                        "NFL_PLASMA": "nfl",
                        "MMSE_BASELINE": "mmse",
                        "APOE4_COUNT": "apoe4_count",
                        "TAU_CSF": "tau_csf",
                        "ABETA42_PLASMA": "abeta42_plasma",
                        "ABETA40_PLASMA": "abeta40_plasma",
                    }
                    patient_data[key_map.get(feat_name, feat_name.lower())] = value

            elif rtype == "Condition":
                # Extract APOE4 from Condition if encoded that way
                pass  # extend if needed

        log.info(
            "InferencePreprocessor.from_fhir_bundle: extracted",
            n_features=len(patient_data),
            keys=sorted(patient_data.keys()),
        )
        return self.from_dict(patient_data, patient_id=patient_id)


# ---------------------------------------------------------------------------
# FHIR parsing helpers
# ---------------------------------------------------------------------------

def _extract_loinc_code(resource: dict) -> str | None:
    """Extract first LOINC code from an Observation.code.

    Args:
        resource: FHIR Observation resource dict.

    Returns:
        LOINC code string or None.
    """
    code_obj = resource.get("code", {})
    # Try coding array first
    for coding in code_obj.get("coding", []):
        system = coding.get("system", "")
        if "loinc" in system.lower():
            return coding.get("code")
    # Try local extension codes (e.g. "NF-PTAU217")
    for coding in code_obj.get("coding", []):
        code = coding.get("code", "")
        if code in LOINC_TO_FEATURE:
            return code
    # Fallback: try text match on local codes
    text = code_obj.get("text", "")
    for key in LOINC_TO_FEATURE:
        if key in text.upper():
            return key
    return None


def _extract_observation_value(resource: dict) -> float | None:
    """Extract numeric value from FHIR Observation.

    Handles valueQuantity, valueString (for APOE4 genotype), and valueInteger.

    Args:
        resource: FHIR Observation resource dict.

    Returns:
        Float value or None if not parseable.
    """
    # valueQuantity
    vq = resource.get("valueQuantity", {})
    if vq:
        try:
            return float(vq.get("value", float("nan")))
        except (TypeError, ValueError):
            pass

    # valueInteger
    vi = resource.get("valueInteger")
    if vi is not None:
        try:
            return float(vi)
        except (TypeError, ValueError):
            pass

    # valueString (e.g. APOE genotype "e3/e4" → 1)
    vs = resource.get("valueString", "")
    if vs:
        lower = vs.lower().strip()
        if lower in APOE4_GENOTYPE_TO_COUNT:
            return float(APOE4_GENOTYPE_TO_COUNT[lower])
        # Try direct numeric
        try:
            return float(vs)
        except ValueError:
            pass

    # valueCodeableConcept text (e.g. APOE4 as coded concept)
    vcc = resource.get("valueCodeableConcept", {})
    if vcc:
        text = vcc.get("text", "")
        lower = text.lower().strip()
        if lower in APOE4_GENOTYPE_TO_COUNT:
            return float(APOE4_GENOTYPE_TO_COUNT[lower])

    return None


def _hash_id(patient_id: str) -> str:
    """SHA-256 hash a patient ID for PHI-safe logging."""
    return hashlib.sha256(patient_id.encode()).hexdigest()[:16]
