"""CSV-backed PyTorch Dataset for NeuroFusion-AD training on real processed data.

Loads patient records from processed CSV files (ADNI or Bio-Hermes-001) and
returns batches compatible with NeuroFusionAD.forward().

Column mapping logic:
    ADNI: fluid (2), acoustic (12), motor (8), clinical (10 = 8 base + 2 derived)
    Bio-Hermes-001: fluid (2), acoustic (10 padded to 12), motor (8 selected from 15),
                    clinical (10 = 4 base + 6 padded/zeroed)

Phase 2B leakage fix: ABETA42_CSF removed from fluid features.
    AMYLOID_POSITIVE = 1 if ABETA42_CSF < 192 — including it was direct leakage (r=-0.86).
    Fluid now uses only plasma biomarkers: PTAU217 and NFL_PLASMA (both available in
    ADNI and Bio-Hermes-001, enabling consistent cross-cohort alignment).
    The CSF-derived ABETA_PTAU_RATIO clinical feature is replaced with the
    plasma Abeta40/42 ratio (ABETA40_PLASMA / ABETA42_PLASMA), which is
    a validated biomarker and does not use the CSF Abeta42 label source.

Median imputation is fit on the train split and applied to val/test splits.
Bio-Hermes-001 acoustic and motor features require separate standardization
(fit on BH train set) because they are not covered by the ADNI scaler.

PHI compliance: Patient IDs (RID / USUBJID) are SHA-256 hashed immediately
on load. No raw identifiers are ever stored or logged.

IEC 62304 Requirement Traceability:
    SRS-001 § 5.4 — Data Loading Requirements
    SRS-001 § 6.1 — PHI Handling Requirements
    SAD-001 § 5.0 — Data Pipeline Architecture
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

import structlog

log = structlog.get_logger(__name__)


def _hash_patient_id(patient_id: str) -> str:
    """Compute SHA-256 hash of a patient ID for PHI-safe storage.

    Args:
        patient_id: Raw patient identifier string (RID or USUBJID).

    Returns:
        Hex-encoded SHA-256 hash of the patient ID.
    """
    return hashlib.sha256(str(patient_id).encode()).hexdigest()


def _to_tensor(values: list[float]) -> torch.Tensor:
    """Convert a Python list of floats to a FloatTensor.

    Uses list conversion (not torch.from_numpy) to avoid NumPy 2.x ABI issues
    with PyTorch 2.1.2.

    Args:
        values: Python list of float values.

    Returns:
        1-D FloatTensor of the same length.
    """
    return torch.tensor(values, dtype=torch.float32)


class NeuroFusionCSVDataset(Dataset):
    """PyTorch Dataset that loads from processed ADNI or Bio-Hermes-001 CSV files.

    Handles:
        - Column-to-encoder mapping (fluid/acoustic/motor/clinical).
        - Median imputation for null values: fit on train split, apply to val/test.
        - Bio-Hermes-001 acoustic/motor re-standardization (fit on BH train only).
        - Derived clinical features (PTAU_TAU_RATIO, ABETA4240_PLASMA_RATIO for ADNI).
        - Padding Bio-Hermes-001 acoustic from 10 to 12 with zeros.
        - Returns batch dict compatible with NeuroFusionAD.forward().
        - SHA-256 hashed patient IDs (PHI safe).

    Attributes:
        csv_path: Path to the source CSV file.
        mode: 'adni' or 'biohermes'.
        imputation_stats: Dict mapping column name to median fill value.
        biohermes_scaler: Fitted StandardScaler for BH acoustic+motor (or None).
        records: List of pre-processed sample dicts.

    Args:
        csv_path: Absolute path to the processed CSV file.
        mode: 'adni' or 'biohermes'. Controls column mapping.
        fit_imputation: If True, compute imputation medians from this CSV and store
            in self.imputation_stats. Use for the train split.
        imputation_stats: Pre-computed imputation medians dict (from the train split).
            Required when fit_imputation=False (i.e., val/test splits).
        biohermes_scaler: Fitted sklearn StandardScaler for BH acoustic+motor columns.
            Only used when mode='biohermes'. If None, raw values are used.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If mode is not 'adni' or 'biohermes'.
        ValueError: If fit_imputation=False and imputation_stats is None.
    """

    # --- ADNI column definitions ---
    # Phase 2B: ABETA42_CSF removed — it is the source of AMYLOID_POSITIVE label
    # (label = 1 if ABETA42_CSF < 192 pg/mL), causing data leakage (r=-0.86).
    # Two plasma biomarkers used instead: pTau217 and NfL. Both available in
    # ADNI and Bio-Hermes-001, enabling consistent cross-cohort feature alignment.
    ADNI_FLUID_COLS = [
        "PTAU217", "NFL_PLASMA",
    ]
    ADNI_ACOUSTIC_COLS = [
        "acoustic_jitter", "acoustic_shimmer", "acoustic_hnr",
        "acoustic_f0_mean", "acoustic_f0_std",
        "acoustic_mfcc1", "acoustic_mfcc2", "acoustic_mfcc3",
        "acoustic_mfcc4", "acoustic_mfcc5", "acoustic_mfcc6", "acoustic_mfcc7",
    ]
    ADNI_MOTOR_COLS = [
        "motor_tremor_freq", "motor_tremor_amp", "motor_bradykinesia_score",
        "motor_spiral_rmse", "motor_tapping_cv", "motor_tapping_asymmetry",
        "motor_grip_force_mean", "motor_grip_force_cv",
    ]
    # 8 base clinical columns; 2 derived columns are computed at runtime
    ADNI_CLINICAL_BASE_COLS = [
        "AGE", "SEX_CODE", "EDUCATION_YEARS", "MMSE_BASELINE",
        "APOE4_COUNT", "TAU_CSF", "ABETA42_PLASMA", "ABETA40_PLASMA",
    ]

    # --- Bio-Hermes-001 column definitions ---
    # Phase 2B: aligned with ADNI — same 2 plasma biomarkers (pTau217 + NfL)
    BH_FLUID_COLS = [
        "PTAU217", "NFL_PLASMA",
    ]
    # 10 acoustic columns; padded to 12 with zeros
    BH_ACOUSTIC_COLS = [
        "acoustic_delayed_recall", "acoustic_object_recall",
        "acoustic_image_descr_score", "acoustic_intraword_pause",
        "acoustic_speaking_rate", "acoustic_verbal_fluency",
        "acoustic_image_speaking_rate", "acoustic_naming_duration",
        "acoustic_monotonicity", "acoustic_pause_rate",
    ]
    # First 8 selected from 15 available motor columns
    BH_MOTOR_COLS = [
        "motor_dcr_clock_score", "motor_dcr_delayed_recall", "motor_dcr_score",
        "motor_sdmt_acc", "motor_sdmt_attempted", "motor_spiral_cw_dom",
        "motor_trails_b_acc", "motor_trails_b_time",
    ]
    BH_CLINICAL_BASE_COLS = ["AGE", "SEX_CODE", "EDUCATION_YEARS", "MMSE_BASELINE", "ABETA40_PLASMA"]

    def __init__(
        self,
        csv_path: str,
        mode: str = "adni",
        fit_imputation: bool = False,
        imputation_stats: dict[str, float] | None = None,
        biohermes_scaler: StandardScaler | None = None,
    ) -> None:
        """Initialize NeuroFusionCSVDataset from a processed CSV file.

        Args:
            csv_path: Absolute path to processed CSV (adni_train.csv, etc.).
            mode: 'adni' or 'biohermes'.
            fit_imputation: Compute imputation medians from this CSV (train split).
            imputation_stats: Pre-fit imputation medians dict (val/test splits).
            biohermes_scaler: Fitted StandardScaler for BH acoustic+motor (mode='biohermes').

        Raises:
            FileNotFoundError: If csv_path does not exist.
            ValueError: If mode invalid, or fit_imputation=False with no stats.
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        if mode not in ("adni", "biohermes"):
            raise ValueError(f"mode must be 'adni' or 'biohermes', got '{mode}'")
        if not fit_imputation and imputation_stats is None:
            raise ValueError(
                "imputation_stats must be provided when fit_imputation=False "
                "(required for val/test splits to use train-fit statistics)."
            )

        self.csv_path = csv_path
        self.mode = mode
        self.biohermes_scaler = biohermes_scaler

        log.info("NeuroFusionCSVDataset loading", csv_path=str(csv_path), mode=mode)
        df = pd.read_csv(str(csv_path))
        log.info("CSV loaded", n_rows=len(df), n_cols=len(df.columns))

        # Replace ADNI sentinel values (-1, -4) with NaN
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].replace([-1, -4], np.nan)

        # Fit or apply imputation stats
        if fit_imputation:
            self.imputation_stats = self._fit_imputation(df)
        else:
            self.imputation_stats = imputation_stats

        df = self._apply_imputation(df)

        # Build per-sample records
        self.records = self._build_records(df)
        log.info(
            "NeuroFusionCSVDataset ready",
            n_samples=len(self.records),
            mode=mode,
        )

    def _fit_imputation(self, df: pd.DataFrame) -> dict[str, float]:
        """Compute column-wise median imputation values from this DataFrame.

        Args:
            df: Source DataFrame (should be the train split).

        Returns:
            Dict mapping column name to median value (0.0 if column absent or all NaN).
        """
        all_cols = (
            self.ADNI_FLUID_COLS + self.ADNI_ACOUSTIC_COLS + self.ADNI_MOTOR_COLS
            + self.ADNI_CLINICAL_BASE_COLS
            if self.mode == "adni"
            else self.BH_FLUID_COLS + self.BH_ACOUSTIC_COLS + self.BH_MOTOR_COLS
            + self.BH_CLINICAL_BASE_COLS
        )
        stats: dict[str, float] = {}
        for col in all_cols:
            if col in df.columns and df[col].notna().any():
                stats[col] = float(df[col].median())
            else:
                stats[col] = 0.0
        log.debug("imputation_stats_fit", n_cols=len(stats))
        return stats

    def _apply_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using pre-fit imputation statistics.

        Args:
            df: DataFrame to impute.

        Returns:
            DataFrame with NaN values replaced by median statistics.
        """
        for col, fill_val in self.imputation_stats.items():
            if col in df.columns:
                df[col] = df[col].fillna(fill_val)
        return df

    def _get_patient_id_col(self, df: pd.DataFrame) -> str:
        """Determine the patient ID column based on mode and available columns.

        Args:
            df: Source DataFrame.

        Returns:
            Column name for the patient ID ('RID' for ADNI, 'USUBJID' for BH).
        """
        if self.mode == "adni" and "RID" in df.columns:
            return "RID"
        if self.mode == "biohermes" and "USUBJID" in df.columns:
            return "USUBJID"
        # Fallback: use index
        return None

    def _safe_col(self, df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
        """Return a column from df, or a constant Series if the column is absent.

        Args:
            df: Source DataFrame.
            col: Column name to retrieve.
            default: Default value to use if column is missing.

        Returns:
            pd.Series with the column values or the default fill.
        """
        if col in df.columns:
            return df[col].fillna(default)
        log.warning("csv_dataset_missing_column", col=col, default=default)
        return pd.Series([default] * len(df), index=df.index)

    def _build_adni_record(self, row: pd.Series) -> dict[str, Any]:
        """Build a single ADNI sample dict from a DataFrame row.

        Args:
            row: Single row from the ADNI processed CSV.

        Returns:
            Dict with 'fluid', 'acoustic', 'motor', 'clinical', label keys,
            and 'patient_id' (SHA-256 hashed).
        """
        # Fluid (2): PTAU217, NFL_PLASMA only (ABETA42_CSF removed — Phase 2B leakage fix)
        fluid = [float(row.get(c, 0.0)) for c in self.ADNI_FLUID_COLS]

        # Acoustic (12)
        acoustic = [float(row.get(c, 0.0)) for c in self.ADNI_ACOUSTIC_COLS]

        # Motor (8)
        motor = [float(row.get(c, 0.0)) for c in self.ADNI_MOTOR_COLS]

        # Clinical (10): 8 base + 2 derived (non-leaking)
        clin_base = [float(row.get(c, 0.0)) for c in self.ADNI_CLINICAL_BASE_COLS]
        tau_csf = float(row.get("TAU_CSF", 0.0))
        ptau181_csf = float(row.get("PTAU181_CSF", 0.0))
        # Derived feature 9: pTau181/totalTau ratio — neurodegeneration marker (non-leaking)
        ptau_tau_ratio = ptau181_csf / (tau_csf + 1e-6)
        # Derived feature 10: plasma Abeta40/42 ratio — validated plasma biomarker
        # Uses ABETA40_PLASMA and ABETA42_PLASMA (already in clin_base positions 7-8).
        # CSF ABETA42_CSF is intentionally excluded here (label source — Phase 2B fix).
        abeta42_plasma = float(row.get("ABETA42_PLASMA", 0.0))
        abeta40_plasma = float(row.get("ABETA40_PLASMA", 0.0))
        abeta4240_plasma_ratio = abeta40_plasma / (abeta42_plasma + 1e-6)
        clinical = clin_base + [ptau_tau_ratio, abeta4240_plasma_ratio]  # 10 total

        # Labels
        amyloid_label = float(row["AMYLOID_POSITIVE"]) if "AMYLOID_POSITIVE" in row.index and not pd.isna(row["AMYLOID_POSITIVE"]) else float("nan")
        mmse_slope = float(row["MMSE_SLOPE"]) if "MMSE_SLOPE" in row.index and not pd.isna(row["MMSE_SLOPE"]) else float("nan")
        time_to_event = float(row["TIME_TO_EVENT"]) if "TIME_TO_EVENT" in row.index and not pd.isna(row["TIME_TO_EVENT"]) else float("nan")
        event_indicator = float(row["EVENT_INDICATOR"]) if "EVENT_INDICATOR" in row.index and not pd.isna(row["EVENT_INDICATOR"]) else float("nan")

        pid = str(row.get("RID", str(row.name)))
        return {
            "fluid": fluid,
            "acoustic": acoustic,
            "motor": motor,
            "clinical": clinical,
            "amyloid_label": amyloid_label,
            "mmse_slope": mmse_slope,
            "survival_time": time_to_event,
            "event_indicator": event_indicator,
            "patient_id": _hash_patient_id(pid),
        }

    def _build_biohermes_record(self, row: pd.Series) -> dict[str, Any]:
        """Build a single Bio-Hermes-001 sample dict from a DataFrame row.

        Bio-Hermes-001 specifics:
            - Acoustic: 10 features padded to 12 with zeros.
            - Motor: 8 of 15 features selected.
            - Clinical: 4 base columns + 6 zeros/derived (no APOE4, TAU_CSF, etc.).
            - Labels: Only AMYLOID_POSITIVE (MMSE_SLOPE and survival are all NaN).

        Args:
            row: Single row from the Bio-Hermes-001 processed CSV.

        Returns:
            Dict with 'fluid', 'acoustic', 'motor', 'clinical', label keys,
            and 'patient_id' (SHA-256 hashed).
        """
        # Fluid (2): PTAU217, NFL_PLASMA only (aligned with ADNI — Phase 2B)
        fluid = [float(row.get(c, 0.0)) for c in self.BH_FLUID_COLS]

        # Acoustic (10 + 2 zeros = 12)
        acoustic_10 = [float(row.get(c, 0.0)) for c in self.BH_ACOUSTIC_COLS]
        acoustic = acoustic_10 + [0.0, 0.0]  # pad to 12

        # Motor (8 selected)
        motor = [float(row.get(c, 0.0)) for c in self.BH_MOTOR_COLS]

        # Apply Bio-Hermes digital scaler if provided
        if self.biohermes_scaler is not None:
            combined = acoustic + motor  # 20 values
            combined_arr = np.array([combined], dtype=np.float64)
            try:
                scaled = self.biohermes_scaler.transform(combined_arr)[0].tolist()
                acoustic = scaled[:12]
                motor = scaled[12:20]
            except Exception as exc:  # pragma: no cover
                log.warning("biohermes_scaler_transform_failed", error=str(exc))

        # Clinical (10): 4 base + 6 padded
        age = float(row.get("AGE", 0.0))
        sex_code = float(row.get("SEX_CODE", 0.0))
        edu_years = float(row.get("EDUCATION_YEARS", 0.0))
        mmse_base = float(row.get("MMSE_BASELINE", 0.0))
        # Positions 5-10: no APOE4, no TAU_CSF, BH plasma Abeta42/40, derived ratio, 0
        abeta42_plasma = float(row.get("ABETA42_PLASMA", 0.0))
        abeta40_plasma = float(row.get("ABETA40_PLASMA", 0.0)) if "ABETA40_PLASMA" in row.index else 0.0
        ptau217 = float(row.get("PTAU217", 0.0))
        abeta4240_ratio = float(row.get("ABETA4240_RATIO", 0.0))
        plasma_ratio = ptau217 / (abeta4240_ratio + 1e-6)
        clinical = [age, sex_code, edu_years, mmse_base, 0.0, 0.0, abeta42_plasma, abeta40_plasma, plasma_ratio, 0.0]

        # Labels: only AMYLOID_POSITIVE is valid for BH
        amyloid_label = float(row["AMYLOID_POSITIVE"]) if "AMYLOID_POSITIVE" in row.index and not pd.isna(row["AMYLOID_POSITIVE"]) else float("nan")

        pid = str(row.get("USUBJID", str(row.name)))
        return {
            "fluid": fluid,
            "acoustic": acoustic,
            "motor": motor,
            "clinical": clinical,
            "amyloid_label": amyloid_label,
            "mmse_slope": float("nan"),      # cross-sectional only
            "survival_time": float("nan"),   # no survival labels
            "event_indicator": float("nan"), # no survival labels
            "patient_id": _hash_patient_id(pid),
        }

    def _build_records(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """Build the full list of sample records from the processed DataFrame.

        Args:
            df: Processed and imputed DataFrame.

        Returns:
            List of sample dicts, one per row.
        """
        records = []
        builder = (
            self._build_adni_record
            if self.mode == "adni"
            else self._build_biohermes_record
        )
        for _, row in df.iterrows():
            try:
                rec = builder(row)
                records.append(rec)
            except Exception as exc:  # pragma: no cover
                log.warning(
                    "csv_dataset_row_build_failed",
                    row_index=str(row.name),
                    error=str(exc),
                )
        return records

    def __len__(self) -> int:
        """Return the number of patient records in the dataset.

        Returns:
            Integer count of valid samples.
        """
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a single patient record as tensors.

        Args:
            idx: Integer index in [0, len(dataset)).

        Returns:
            Dict with keys:
                - 'fluid': FloatTensor [2]  # Phase 2B: PTAU217 + NFL_PLASMA only
                - 'acoustic': FloatTensor [12]
                - 'motor': FloatTensor [8]
                - 'clinical': FloatTensor [10]
                - 'amyloid_label': FloatTensor scalar (may be NaN)
                - 'mmse_slope': FloatTensor scalar (may be NaN)
                - 'survival_time': FloatTensor scalar (may be NaN)
                - 'event_indicator': FloatTensor scalar (may be NaN)
                - 'patient_id': str (SHA-256 hash, never raw PHI)

        Raises:
            IndexError: If idx is out of bounds.
        """
        if idx < 0 or idx >= len(self.records):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of size {len(self.records)}."
            )
        rec = self.records[idx]
        return {
            "fluid": _to_tensor(rec["fluid"]),
            "acoustic": _to_tensor(rec["acoustic"]),
            "motor": _to_tensor(rec["motor"]),
            "clinical": _to_tensor(rec["clinical"]),
            "amyloid_label": torch.tensor(rec["amyloid_label"], dtype=torch.float32),
            "mmse_slope": torch.tensor(rec["mmse_slope"], dtype=torch.float32),
            "survival_time": torch.tensor(rec["survival_time"], dtype=torch.float32),
            "event_indicator": torch.tensor(rec["event_indicator"], dtype=torch.float32),
            "patient_id": rec["patient_id"],
        }

    @classmethod
    def fit_biohermes_scaler(
        cls,
        bh_train_csv_path: str,
        save_path: str,
    ) -> StandardScaler:
        """Fit and save a StandardScaler on Bio-Hermes-001 train acoustic+motor features.

        This scaler must be fit on the BH train split ONLY and then applied to
        val/test splits. Saved as a pickle file at save_path.

        Args:
            bh_train_csv_path: Path to biohermes001_train.csv.
            save_path: Destination path for the pickle file (e.g.,
                data/processed/biohermes/biohermes_digital_scaler.pkl).

        Returns:
            Fitted StandardScaler covering [acoustic_cols (12) + motor_cols (8)] = 20 features.
        """
        bh_train_csv_path = Path(bh_train_csv_path)
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        log.info("Fitting BH digital scaler", path=str(bh_train_csv_path))
        df = pd.read_csv(str(bh_train_csv_path))
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].replace([-1, -4], np.nan)

        # Acoustic (10 cols + 2 zero-pads = 12 cols total)
        acoustic_cols = cls.BH_ACOUSTIC_COLS
        acoustic_data = np.zeros((len(df), 12), dtype=np.float64)
        for i, col in enumerate(acoustic_cols):
            if col in df.columns:
                vals = df[col].fillna(df[col].median() if df[col].notna().any() else 0.0)
                acoustic_data[:, i] = vals.values
        # columns 10 and 11 stay zero (padding)

        # Motor (8 cols)
        motor_cols = cls.BH_MOTOR_COLS
        motor_data = np.zeros((len(df), 8), dtype=np.float64)
        for i, col in enumerate(motor_cols):
            if col in df.columns:
                vals = df[col].fillna(df[col].median() if df[col].notna().any() else 0.0)
                motor_data[:, i] = vals.values

        combined = np.hstack([acoustic_data, motor_data])  # [N, 20]

        scaler = StandardScaler()
        scaler.fit(combined)

        with open(save_path, "wb") as f:
            pickle.dump(scaler, f)

        log.info("BH digital scaler saved", path=str(save_path))
        return scaler
