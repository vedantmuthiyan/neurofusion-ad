"""Modality encoders for NeuroFusion-AD.

This module implements four modality-specific encoders that transform
heterogeneous patient data into a common 768-dimensional embedding space
for downstream cross-modal fusion and GNN processing.

All encoders are IEC 62304 Class B compliant — every public method carries
a Google-style docstring and full type annotations. Input validation ensures
physiologically impossible values are rejected before inference.

Encoder inventory:
    FluidBiomarkerEncoder    — 6 plasma/CSF biomarker features → R^768
    DigitalAcousticEncoder   — 12 vocal/acoustic features → R^768
    DigitalMotorEncoder      — 8 motor assessment features → R^768
    ClinicalDemographicEncoder — 10 clinical/demographic features → R^768

Document traceability:
    SRS-001 § 4.2 — Multimodal Input Specification
    SAD-001 § 5.1 — Modality Encoder Architecture
"""

from __future__ import annotations

import torch
import torch.nn as nn
import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Validated input range constants (SRS-001 § 4.2.1)
# ---------------------------------------------------------------------------

# FluidBiomarker feature indices
_FLUID_IDX_PTAU217 = 0
_FLUID_IDX_ABETA_RATIO = 1
_FLUID_IDX_NFL = 2

_FLUID_RANGES: dict[int, tuple[float, float]] = {
    _FLUID_IDX_PTAU217: (0.1, 100.0),       # pTau-217 (pg/mL)
    _FLUID_IDX_ABETA_RATIO: (0.01, 0.30),   # Abeta42/40 ratio
    _FLUID_IDX_NFL: (5.0, 200.0),            # NfL (pg/mL)
}

_FLUID_RANGE_NAMES: dict[int, str] = {
    _FLUID_IDX_PTAU217: "pTau-217",
    _FLUID_IDX_ABETA_RATIO: "Abeta42/40 ratio",
    _FLUID_IDX_NFL: "NfL",
}

# DigitalAcoustic feature indices
_ACOUSTIC_IDX_JITTER = 0
_ACOUSTIC_IDX_SHIMMER = 1

_ACOUSTIC_RANGES: dict[int, tuple[float, float]] = {
    _ACOUSTIC_IDX_JITTER: (0.0001, 0.05),   # jitter (dimensionless)
    _ACOUSTIC_IDX_SHIMMER: (0.001, 0.30),   # shimmer (dimensionless)
}

_ACOUSTIC_RANGE_NAMES: dict[int, str] = {
    _ACOUSTIC_IDX_JITTER: "jitter",
    _ACOUSTIC_IDX_SHIMMER: "shimmer",
}

# ClinicalDemographic feature indices
_CLINICAL_IDX_MMSE = 3

_CLINICAL_RANGES: dict[int, tuple[float, float]] = {
    _CLINICAL_IDX_MMSE: (0.0, 30.0),        # MMSE baseline score
}

_CLINICAL_RANGE_NAMES: dict[int, str] = {
    _CLINICAL_IDX_MMSE: "MMSE",
}


# ---------------------------------------------------------------------------
# Shared validation helper
# ---------------------------------------------------------------------------

def _validate_features(
    x: torch.Tensor,
    ranges: dict[int, tuple[float, float]],
    names: dict[int, str],
    encoder_name: str,
    skip_range_check: bool = False,
) -> None:
    """Validate that feature tensor values are within physiological ranges.

    Args:
        x: Input tensor of shape [batch_size, num_features].
        ranges: Mapping from feature index to (min_val, max_val) tuple.
        names: Mapping from feature index to human-readable feature name.
        encoder_name: Name of the calling encoder, used in error messages.
        skip_range_check: If True, skip physiological range checks (used during
            training with StandardScaler-normalized inputs where raw ranges
            do not apply). NaN check is always performed.

    Raises:
        ValueError: If any feature contains NaN values or (when
            skip_range_check is False) values outside the defined
            physiological range for that feature.
    """
    if torch.isnan(x).any():
        raise ValueError(
            f"[{encoder_name}] Input contains NaN values. "
            "Ensure all biomarker measurements are valid before calling encode()."
        )

    if skip_range_check:
        return

    for idx, (lo, hi) in ranges.items():
        feature_col = x[:, idx]
        if (feature_col < lo).any() or (feature_col > hi).any():
            bad_vals = feature_col[
                (feature_col < lo) | (feature_col > hi)
            ].tolist()
            raise ValueError(
                f"[{encoder_name}] Feature '{names[idx]}' (column {idx}) "
                f"contains values outside the valid physiological range "
                f"[{lo}, {hi}]. Offending values: {bad_vals}. "
                "Reject or impute these measurements before inference."
            )


# ---------------------------------------------------------------------------
# Shared MLP block builder
# ---------------------------------------------------------------------------

def _build_encoder_layers(in_features: int) -> nn.Sequential:
    """Construct the shared three-block MLP architecture for all encoders.

    All modality encoders share the same projection topology:
        Block 1: Linear(in, 256)  → LayerNorm(256)  → GELU
        Block 2: Linear(256, 512) → LayerNorm(512)  → GELU → Dropout(0.1)
        Block 3: Linear(512, 768) → LayerNorm(768)

    The final LayerNorm stabilises the 768-dim output for dot-product
    attention in the downstream CrossModalAttention module.

    Args:
        in_features: Number of input features for the first linear layer.

    Returns:
        An nn.Sequential containing all layers in order.
    """
    return nn.Sequential(
        # Block 1
        nn.Linear(in_features, 256),
        nn.LayerNorm(256),
        nn.GELU(),
        # Block 2
        nn.Linear(256, 512),
        nn.LayerNorm(512),
        nn.GELU(),
        nn.Dropout(p=0.1),
        # Block 3
        nn.Linear(512, 768),
        nn.LayerNorm(768),
    )


# ---------------------------------------------------------------------------
# FluidBiomarkerEncoder
# ---------------------------------------------------------------------------

class FluidBiomarkerEncoder(nn.Module):
    """Encodes plasma/CSF fluid biomarker panels into a 768-dim embedding.

    Transforms a six-feature vector of Alzheimer's-relevant biomarkers
    (pTau-217, Abeta42/40 ratio, NfL, GFAP, total-tau, Abeta42) into a
    normalised 768-dimensional representation suitable for cross-modal fusion.

    Input feature schema (SRS-001 § 4.2.1):
        index 0 — pTau-217 (pg/mL),      valid range: [0.1, 100]
        index 1 — Abeta42/40 ratio,       valid range: [0.01, 0.30]
        index 2 — NfL (pg/mL),            valid range: [5, 200]
        index 3 — GFAP (pg/mL),           no range check
        index 4 — total-tau (pg/mL),      no range check
        index 5 — Abeta42 (pg/mL),        no range check

    Architecture:
        Linear(6, 256) → LayerNorm(256) → GELU →
        Linear(256, 512) → LayerNorm(512) → GELU → Dropout(0.1) →
        Linear(512, 768) → LayerNorm(768)

    Example:
        >>> encoder = FluidBiomarkerEncoder()
        >>> x = torch.zeros(4, 6)
        >>> x[:, 0] = 5.0   # pTau-217
        >>> x[:, 1] = 0.1   # Abeta42/40
        >>> x[:, 2] = 50.0  # NfL
        >>> out = encoder(x)
        >>> out.shape
        torch.Size([4, 768])

    Raises:
        ValueError: If input contains NaN values or any validated biomarker
            falls outside its defined physiological range.
    """

    INPUT_DIM: int = 6
    OUTPUT_DIM: int = 768

    def __init__(self) -> None:
        """Initialise the FluidBiomarkerEncoder layers."""
        super().__init__()
        self.net = _build_encoder_layers(self.INPUT_DIM)
        log.info(
            "FluidBiomarkerEncoder initialised",
            input_dim=self.INPUT_DIM,
            output_dim=self.OUTPUT_DIM,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode fluid biomarker features into a 768-dim embedding.

        Args:
            x: Float tensor of shape [batch_size, 6] containing the six
                fluid biomarker measurements for each sample in the batch.

        Returns:
            Float tensor of shape [batch_size, 768] — the normalised
            biomarker embedding ready for cross-modal attention.

        Raises:
            ValueError: If input contains NaN values or validated biomarkers
                (pTau-217, Abeta42/40, NfL) fall outside physiological ranges.
        """
        _validate_features(
            x,
            ranges=_FLUID_RANGES,
            names=_FLUID_RANGE_NAMES,
            encoder_name="FluidBiomarkerEncoder",
            skip_range_check=self.training,
        )
        embedding: torch.Tensor = self.net(x)
        log.debug(
            "FluidBiomarkerEncoder forward complete",
            batch_size=x.shape[0],
            output_shape=list(embedding.shape),
        )
        return embedding


# ---------------------------------------------------------------------------
# DigitalAcousticEncoder
# ---------------------------------------------------------------------------

class DigitalAcousticEncoder(nn.Module):
    """Encodes vocal/acoustic digital biomarker features into a 768-dim embedding.

    Transforms a twelve-feature vector capturing voice perturbation and
    spectral properties (jitter, shimmer, HNR, F0 statistics, MFCCs) into
    a normalised 768-dimensional representation for cross-modal fusion.

    Input feature schema (SRS-001 § 4.2.2):
        index  0 — jitter (dimensionless),    valid range: [0.0001, 0.05]
        index  1 — shimmer (dimensionless),   valid range: [0.001, 0.30]
        index  2 — HNR (dB),                  no range check
        index  3 — F0_mean (Hz),              no range check
        index  4 — F0_std (Hz),               no range check
        index  5 — MFCC_1                     no range check
        index  6 — MFCC_2                     no range check
        index  7 — MFCC_3                     no range check
        index  8 — MFCC_4                     no range check
        index  9 — MFCC_5                     no range check
        index 10 — MFCC_6                     no range check
        index 11 — MFCC_7                     no range check

    Architecture:
        Linear(12, 256) → LayerNorm(256) → GELU →
        Linear(256, 512) → LayerNorm(512) → GELU → Dropout(0.1) →
        Linear(512, 768) → LayerNorm(768)

    Example:
        >>> encoder = DigitalAcousticEncoder()
        >>> x = torch.zeros(4, 12)
        >>> x[:, 0] = 0.005   # jitter
        >>> x[:, 1] = 0.05    # shimmer
        >>> out = encoder(x)
        >>> out.shape
        torch.Size([4, 768])

    Raises:
        ValueError: If input contains NaN values or jitter/shimmer fall
            outside their defined physiological ranges.
    """

    INPUT_DIM: int = 12
    OUTPUT_DIM: int = 768

    def __init__(self) -> None:
        """Initialise the DigitalAcousticEncoder layers."""
        super().__init__()
        self.net = _build_encoder_layers(self.INPUT_DIM)
        log.info(
            "DigitalAcousticEncoder initialised",
            input_dim=self.INPUT_DIM,
            output_dim=self.OUTPUT_DIM,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode acoustic digital biomarker features into a 768-dim embedding.

        Args:
            x: Float tensor of shape [batch_size, 12] containing the twelve
                acoustic feature measurements for each sample in the batch.

        Returns:
            Float tensor of shape [batch_size, 768] — the normalised
            acoustic embedding ready for cross-modal attention.

        Raises:
            ValueError: If input contains NaN values or jitter/shimmer fall
                outside their physiological ranges.
        """
        _validate_features(
            x,
            ranges=_ACOUSTIC_RANGES,
            names=_ACOUSTIC_RANGE_NAMES,
            encoder_name="DigitalAcousticEncoder",
            skip_range_check=self.training,
        )
        embedding: torch.Tensor = self.net(x)
        log.debug(
            "DigitalAcousticEncoder forward complete",
            batch_size=x.shape[0],
            output_shape=list(embedding.shape),
        )
        return embedding


# ---------------------------------------------------------------------------
# DigitalMotorEncoder
# ---------------------------------------------------------------------------

class DigitalMotorEncoder(nn.Module):
    """Encodes motor assessment digital biomarker features into a 768-dim embedding.

    Transforms an eight-feature vector derived from wearable and tablet-based
    motor assessments (tremor, bradykinesia, spiral drawing, finger tapping,
    grip force) into a normalised 768-dimensional representation.

    Input feature schema (SRS-001 § 4.2.3):
        index 0 — tremor_freq (Hz)
        index 1 — tremor_amp (m/s²)
        index 2 — bradykinesia_score (dimensionless)
        index 3 — spiral_RMSE (pixels or normalised units)
        index 4 — tapping_interval_cv (coefficient of variation)
        index 5 — tapping_asymmetry (dimensionless)
        index 6 — grip_force_mean (N)
        index 7 — grip_force_cv (coefficient of variation)

    No physiological range validation is applied for motor features; the
    diversity of assessment devices and normalisation schemes makes hard
    cutoffs impractical at this stage (see SAD-001 § 5.1.3).

    Architecture:
        Linear(8, 256) → LayerNorm(256) → GELU →
        Linear(256, 512) → LayerNorm(512) → GELU → Dropout(0.1) →
        Linear(512, 768) → LayerNorm(768)

    Example:
        >>> encoder = DigitalMotorEncoder()
        >>> x = torch.randn(4, 8)
        >>> out = encoder(x)
        >>> out.shape
        torch.Size([4, 768])

    Raises:
        ValueError: If input contains NaN values.
    """

    INPUT_DIM: int = 8
    OUTPUT_DIM: int = 768

    def __init__(self) -> None:
        """Initialise the DigitalMotorEncoder layers."""
        super().__init__()
        self.net = _build_encoder_layers(self.INPUT_DIM)
        log.info(
            "DigitalMotorEncoder initialised",
            input_dim=self.INPUT_DIM,
            output_dim=self.OUTPUT_DIM,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode motor digital biomarker features into a 768-dim embedding.

        Args:
            x: Float tensor of shape [batch_size, 8] containing the eight
                motor assessment measurements for each sample in the batch.

        Returns:
            Float tensor of shape [batch_size, 768] — the normalised
            motor embedding ready for cross-modal attention.

        Raises:
            ValueError: If input contains NaN values.
        """
        if torch.isnan(x).any():
            raise ValueError(
                "[DigitalMotorEncoder] Input contains NaN values. "
                "Ensure all motor measurements are valid before calling forward()."
            )
        embedding: torch.Tensor = self.net(x)
        log.debug(
            "DigitalMotorEncoder forward complete",
            batch_size=x.shape[0],
            output_shape=list(embedding.shape),
        )
        return embedding


# ---------------------------------------------------------------------------
# ClinicalDemographicEncoder
# ---------------------------------------------------------------------------

class ClinicalDemographicEncoder(nn.Module):
    """Encodes clinical and demographic features into a 768-dim embedding.

    Transforms a ten-feature vector of standard clinical assessments and
    demographic variables into a normalised 768-dimensional representation
    for cross-modal fusion.

    Input feature schema (SRS-001 § 4.2.4):
        index 0 — age (years)
        index 1 — education_years (years)
        index 2 — sex (0 = female, 1 = male)
        index 3 — MMSE_baseline (score 0–30),  valid range: [0, 30]
        index 4 — CDR_sum_boxes (score)
        index 5 — GDS (Geriatric Depression Scale score)
        index 6 — BMI (kg/m²)
        index 7 — systolic_BP (mmHg)
        index 8 — has_APOE4 (0 = no, 1 = yes)
        index 9 — comorbidity_count (integer count)

    Architecture:
        Linear(10, 256) → LayerNorm(256) → GELU →
        Linear(256, 512) → LayerNorm(512) → GELU → Dropout(0.1) →
        Linear(512, 768) → LayerNorm(768)

    Example:
        >>> encoder = ClinicalDemographicEncoder()
        >>> x = torch.zeros(4, 10)
        >>> x[:, 3] = 25.0   # MMSE baseline
        >>> out = encoder(x)
        >>> out.shape
        torch.Size([4, 768])

    Raises:
        ValueError: If input contains NaN values or the MMSE_baseline score
            falls outside [0, 30].
    """

    INPUT_DIM: int = 10
    OUTPUT_DIM: int = 768

    def __init__(self) -> None:
        """Initialise the ClinicalDemographicEncoder layers."""
        super().__init__()
        self.net = _build_encoder_layers(self.INPUT_DIM)
        log.info(
            "ClinicalDemographicEncoder initialised",
            input_dim=self.INPUT_DIM,
            output_dim=self.OUTPUT_DIM,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode clinical and demographic features into a 768-dim embedding.

        Args:
            x: Float tensor of shape [batch_size, 10] containing the ten
                clinical/demographic measurements for each sample in the batch.

        Returns:
            Float tensor of shape [batch_size, 768] — the normalised
            clinical embedding ready for cross-modal attention.

        Raises:
            ValueError: If input contains NaN values or MMSE_baseline
                (index 3) falls outside [0, 30].
        """
        _validate_features(
            x,
            ranges=_CLINICAL_RANGES,
            names=_CLINICAL_RANGE_NAMES,
            encoder_name="ClinicalDemographicEncoder",
            skip_range_check=self.training,
        )
        embedding: torch.Tensor = self.net(x)
        log.debug(
            "ClinicalDemographicEncoder forward complete",
            batch_size=x.shape[0],
            output_shape=list(embedding.shape),
        )
        return embedding
