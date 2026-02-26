"""ADNI dataset preprocessing for NeuroFusion-AD.

Handles normalization, missing value imputation, and batched record processing
for ADNI patient records. Real ADNI data requires a Data Use Agreement (DUA)
from adni.loni.usc.edu — this module assumes data has been obtained legally.

IEC 62304 Requirement Traceability: SRS-001 § 5.1 (Data Preprocessing)
"""

import structlog
import torch
import numpy as np
from typing import Any

logger = structlog.get_logger(__name__)

# Feature dimension constants
_FLUID_DIM = 6
_ACOUSTIC_DIM = 12
_MOTOR_DIM = 8
_CLINICAL_DIM = 10

# Normalization clip bounds
_CLIP_MIN = -5.0
_CLIP_MAX = 5.0


class ADNIPreprocessor:
    """Preprocesses ADNI dataset records into NeuroFusionAD model inputs.

    Handles missing value imputation, feature normalization (z-score), and
    data quality checks. Population-level statistics (mean/std) are estimated
    from published literature and will be updated once real ADNI data is
    obtained under DUA.

    ADNI data must be obtained via Data Use Agreement from adni.loni.usc.edu.
    No real patient data is stored in this repository.

    Attributes:
        missing_strategy: Strategy for imputing missing values ("mean", "median", "zero").

    IEC 62304 Traceability:
        SRS-001 § 5.1 — Input preprocessing requirements
        SDP-001 § 6.2 — Data normalization specification
    """

    # Feature means and stds for normalization (population-level estimates).
    # Order: ptau217, abeta42_40_ratio, nfl, gfap, total_tau, abeta42
    FLUID_MEAN = torch.tensor([12.0, 0.10, 30.0, 150.0, 250.0, 800.0])
    FLUID_STD  = torch.tensor([15.0, 0.05, 20.0,  80.0, 100.0, 200.0])

    # Order: jitter, shimmer, hnr, f0_mean, f0_std, mfcc_1..7
    ACOUSTIC_MEAN = torch.tensor([0.005, 0.04, 15.0, 130.0, 25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ACOUSTIC_STD  = torch.tensor([0.005, 0.02,  5.0,  30.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Order: tremor_freq, tremor_amp, bradykinesia_score, spiral_rmse,
    #        tapping_interval_cv, tapping_asymmetry, grip_force_mean, grip_force_cv
    MOTOR_MEAN = torch.tensor([4.0, 0.3, 50.0, 2.0, 0.15, 0.05, 25.0, 0.10])
    MOTOR_STD  = torch.tensor([2.0, 0.2, 20.0, 1.0, 0.05, 0.03, 10.0, 0.05])

    # Order: age, education, sex, mmse, cdr_sum, gds, bmi, systolic_bp, apoe4, comorbidities
    CLINICAL_MEAN = torch.tensor([72.0, 14.0, 0.5, 26.0, 1.5, 5.0, 27.0, 130.0, 0.3, 2.0])
    CLINICAL_STD  = torch.tensor([ 8.0,  3.0, 0.5,  3.0, 1.5, 3.0,  5.0,  15.0, 0.45, 1.5])

    _VALID_STRATEGIES = ("mean", "median", "zero")

    def __init__(self, missing_strategy: str = "mean") -> None:
        """Initialize ADNIPreprocessor.

        Args:
            missing_strategy: How to handle missing (NaN) values.
                One of "mean" (replace with feature mean), "median" (replace
                with feature median), or "zero" (replace with 0.0).

        Raises:
            ValueError: If missing_strategy is not one of the valid options.
        """
        if missing_strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"missing_strategy must be one of {self._VALID_STRATEGIES}, "
                f"got '{missing_strategy}'"
            )
        self.missing_strategy = missing_strategy
        logger.info("adni_preprocessor_initialized", missing_strategy=missing_strategy)

    def normalize(
        self,
        features: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """Apply z-score normalization and clip to [-5, 5].

        Computes (features - mean) / std and clips the result to prevent
        extreme outliers from destabilizing model training.

        Args:
            features: Input tensor of shape [feature_dim] or [batch, feature_dim].
            mean: Per-feature mean tensor of shape [feature_dim].
            std: Per-feature standard deviation tensor of shape [feature_dim].

        Returns:
            Normalized tensor of the same shape as features, clipped to [-5, 5].
        """
        # Guard against zero std to prevent division by zero
        safe_std = std.clone()
        safe_std[safe_std < 1e-8] = 1e-8

        normalized = (features - mean) / safe_std
        return torch.clamp(normalized, _CLIP_MIN, _CLIP_MAX)

    def impute_missing(
        self,
        features: torch.Tensor,
        mean: torch.Tensor,
    ) -> torch.Tensor:
        """Replace NaN values with column-wise statistics.

        The imputation strategy is determined by self.missing_strategy:
        - "mean": Replace NaN with the provided population mean.
        - "median": Replace NaN with the per-column median of non-NaN values.
        - "zero": Replace NaN with 0.0.

        Args:
            features: Input tensor of shape [feature_dim] or [batch, feature_dim].
            mean: Per-feature population mean of shape [feature_dim], used
                when missing_strategy is "mean" or as fallback when all
                values in a column are NaN.

        Returns:
            Tensor of same shape as features with no NaN values.
        """
        was_1d = features.dim() == 1
        if was_1d:
            features = features.unsqueeze(0)

        result = features.clone()
        nan_mask = torch.isnan(result)

        if not nan_mask.any():
            return result.squeeze(0) if was_1d else result

        if self.missing_strategy == "mean":
            fill_values = mean
        elif self.missing_strategy == "median":
            fill_values = torch.zeros(features.shape[1])
            for col_idx in range(features.shape[1]):
                col = features[:, col_idx]
                valid = col[~torch.isnan(col)]
                if valid.numel() > 0:
                    fill_values[col_idx] = valid.median()
                else:
                    fill_values[col_idx] = mean[col_idx]
        else:  # "zero"
            fill_values = torch.zeros(features.shape[1])

        for col_idx in range(features.shape[1]):
            col_mask = nan_mask[:, col_idx]
            if col_mask.any():
                result[col_mask, col_idx] = fill_values[col_idx]

        n_imputed = nan_mask.sum().item()
        logger.debug(
            "missing_values_imputed",
            strategy=self.missing_strategy,
            n_imputed=int(n_imputed),
        )

        return result.squeeze(0) if was_1d else result

    def preprocess_record(self, record: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Process a single ADNI patient record into normalized tensors.

        Converts raw numpy arrays or lists into float32 tensors, imputes
        missing values, and applies z-score normalization.

        Args:
            record: Dict with keys 'fluid', 'acoustic', 'motor', 'clinical',
                each containing a raw numpy array, list, or torch.Tensor of
                the appropriate feature dimension.

        Returns:
            Dict with keys 'fluid', 'acoustic', 'motor', 'clinical', each
            containing a normalized float32 torch.Tensor of shape [feature_dim]:
                - 'fluid': Tensor[6]
                - 'acoustic': Tensor[12]
                - 'motor': Tensor[8]
                - 'clinical': Tensor[10]

        Raises:
            KeyError: If any required key is missing from record.
            ValueError: If any feature array has incorrect length.
        """
        required_keys = ("fluid", "acoustic", "motor", "clinical")
        for key in required_keys:
            if key not in record:
                raise KeyError(f"Record is missing required key: '{key}'")

        expected_dims = {
            "fluid": _FLUID_DIM,
            "acoustic": _ACOUSTIC_DIM,
            "motor": _MOTOR_DIM,
            "clinical": _CLINICAL_DIM,
        }
        means = {
            "fluid": self.FLUID_MEAN,
            "acoustic": self.ACOUSTIC_MEAN,
            "motor": self.MOTOR_MEAN,
            "clinical": self.CLINICAL_MEAN,
        }
        stds = {
            "fluid": self.FLUID_STD,
            "acoustic": self.ACOUSTIC_STD,
            "motor": self.MOTOR_STD,
            "clinical": self.CLINICAL_STD,
        }

        processed: dict[str, torch.Tensor] = {}
        for key in required_keys:
            raw = record[key]
            if isinstance(raw, torch.Tensor):
                tensor = raw.float()
            elif hasattr(raw, "tolist"):
                # numpy array or similar — convert via .tolist() to avoid
                # torch.from_numpy() incompatibility with NumPy 2.x
                tensor = torch.tensor(raw.tolist(), dtype=torch.float32)
            else:
                tensor = torch.tensor(list(raw), dtype=torch.float32)

            expected = expected_dims[key]
            if tensor.shape[-1] != expected:
                raise ValueError(
                    f"Feature '{key}' has {tensor.shape[-1]} dimensions, "
                    f"expected {expected}."
                )

            tensor = self.impute_missing(tensor, means[key])
            tensor = self.normalize(tensor, means[key], stds[key])
            processed[key] = tensor

        logger.debug("record_preprocessed")
        return processed

    def preprocess_batch(self, records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Process a list of ADNI patient records into batched tensors.

        Calls preprocess_record on each element and stacks results along
        the batch dimension.

        Args:
            records: List of record dicts, each conforming to the format
                expected by preprocess_record.

        Returns:
            Dict with keys 'fluid', 'acoustic', 'motor', 'clinical', each
            containing a float32 torch.Tensor of shape [batch, feature_dim]:
                - 'fluid': Tensor[batch, 6]
                - 'acoustic': Tensor[batch, 12]
                - 'motor': Tensor[batch, 8]
                - 'clinical': Tensor[batch, 10]

        Raises:
            ValueError: If records is empty.
        """
        if not records:
            raise ValueError("records list must not be empty.")

        processed_records = [self.preprocess_record(r) for r in records]
        keys = ("fluid", "acoustic", "motor", "clinical")
        batched: dict[str, torch.Tensor] = {
            key: torch.stack([rec[key] for rec in processed_records], dim=0)
            for key in keys
        }
        logger.info("batch_preprocessed", n_records=len(records))
        return batched
