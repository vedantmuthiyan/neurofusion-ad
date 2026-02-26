"""Unit tests for NeuroFusion-AD modality encoders.

Tests are self-contained — only torch is used for synthetic data generation.
No real patient data (PHI) is used in any test.

Test coverage per encoder:
    - output_shape   : output is [batch_size, 768]
    - output_type    : output dtype is torch.float32
    - valid_input    : valid synthetic data passes without exception
    - invalid_input  : out-of-range values raise ValueError
    - no_nan_output  : output tensor contains no NaN values

Document traceability:
    SRS-001 § 4.2 — Multimodal Input Specification
    SAD-001 § 5.1 — Modality Encoder Architecture
    IEC 62304 § 5.5.5 — Software unit verification
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.encoders import (
    ClinicalDemographicEncoder,
    DigitalAcousticEncoder,
    DigitalMotorEncoder,
    FluidBiomarkerEncoder,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
BATCH_SIZE = 4
EXPECTED_OUTPUT_DIM = 768


# ---------------------------------------------------------------------------
# Synthetic data factories
# (all values are mid-range to avoid validation failures in "valid" tests)
# Pure torch — no numpy to avoid NumPy 2.x / torch ABI incompatibilities.
# ---------------------------------------------------------------------------

def _make_fluid_valid(batch_size: int = BATCH_SIZE) -> torch.Tensor:
    """Return a valid FluidBiomarkerEncoder input tensor using pure torch.

    Six biomarker values are set to physiologically plausible mid-range values:
        index 0 — pTau-217 = 10.0 pg/mL   (range 0.1–100)
        index 1 — Abeta42/40 = 0.15        (range 0.01–0.30)
        index 2 — NfL = 50.0 pg/mL         (range 5–200)
        index 3 — GFAP = 100.0 pg/mL       (unconstrained)
        index 4 — total-tau = 200.0 pg/mL  (unconstrained)
        index 5 — Abeta42 = 800.0 pg/mL    (unconstrained)

    Args:
        batch_size: Number of samples in the batch.

    Returns:
        Float32 tensor of shape [batch_size, 6].
    """
    row = torch.tensor([10.0, 0.15, 50.0, 100.0, 200.0, 800.0], dtype=torch.float32)
    return row.unsqueeze(0).expand(batch_size, -1).clone()


def _make_acoustic_valid(batch_size: int = BATCH_SIZE) -> torch.Tensor:
    """Return a valid DigitalAcousticEncoder input tensor using pure torch.

    Jitter and shimmer are set to mid-range; other features are zero.
        index  0 — jitter  = 0.005  (range 0.0001–0.05)
        index  1 — shimmer = 0.05   (range 0.001–0.30)
        index  2 — HNR     = 15.0 dB
        index  3 — F0_mean = 120.0 Hz
        index  4 — F0_std  = 10.0 Hz
        index 5..11 — MFCC_1..7 = 0.0

    Args:
        batch_size: Number of samples in the batch.

    Returns:
        Float32 tensor of shape [batch_size, 12].
    """
    row = torch.zeros(12, dtype=torch.float32)
    row[0] = 0.005   # jitter
    row[1] = 0.05    # shimmer
    row[2] = 15.0    # HNR
    row[3] = 120.0   # F0_mean
    row[4] = 10.0    # F0_std
    return row.unsqueeze(0).expand(batch_size, -1).clone()


def _make_motor_valid(batch_size: int = BATCH_SIZE) -> torch.Tensor:
    """Return a valid DigitalMotorEncoder input tensor using pure torch.

    Eight motor features are set to representative mid-range values.

    Args:
        batch_size: Number of samples in the batch.

    Returns:
        Float32 tensor of shape [batch_size, 8].
    """
    row = torch.tensor(
        [4.5, 0.02, 2.0, 3.5, 0.15, 0.08, 25.0, 0.12],
        dtype=torch.float32,
    )
    return row.unsqueeze(0).expand(batch_size, -1).clone()


def _make_clinical_valid(batch_size: int = BATCH_SIZE) -> torch.Tensor:
    """Return a valid ClinicalDemographicEncoder input tensor using pure torch.

    MMSE is set to 25 (well within [0, 30]).
        index 0 — age = 70
        index 1 — education_years = 16
        index 2 — sex = 0 (female)
        index 3 — MMSE_baseline = 25
        index 4 — CDR_sum_boxes = 1.5
        index 5 — GDS = 5
        index 6 — BMI = 26
        index 7 — systolic_BP = 125
        index 8 — has_APOE4 = 0
        index 9 — comorbidity_count = 2

    Args:
        batch_size: Number of samples in the batch.

    Returns:
        Float32 tensor of shape [batch_size, 10].
    """
    row = torch.tensor(
        [70.0, 16.0, 0.0, 25.0, 1.5, 5.0, 26.0, 125.0, 0.0, 2.0],
        dtype=torch.float32,
    )
    return row.unsqueeze(0).expand(batch_size, -1).clone()


# ===========================================================================
# FluidBiomarkerEncoder tests
# ===========================================================================

class TestFluidBiomarkerEncoder:
    """Unit tests for FluidBiomarkerEncoder (SRS-001 § 4.2.1)."""

    @pytest.fixture
    def encoder(self) -> FluidBiomarkerEncoder:
        """Provide a freshly initialised FluidBiomarkerEncoder in eval mode.

        Returns:
            FluidBiomarkerEncoder instance set to eval mode.
        """
        model = FluidBiomarkerEncoder()
        model.eval()
        return model

    def test_fluid_output_shape(self, encoder: FluidBiomarkerEncoder) -> None:
        """Output tensor must have shape [batch_size, 768].

        Args:
            encoder: FluidBiomarkerEncoder fixture.
        """
        x = _make_fluid_valid()
        with torch.no_grad():
            out = encoder(x)
        assert out.shape == (BATCH_SIZE, EXPECTED_OUTPUT_DIM), (
            f"Expected shape ({BATCH_SIZE}, {EXPECTED_OUTPUT_DIM}), got {out.shape}"
        )

    def test_fluid_output_type(self, encoder: FluidBiomarkerEncoder) -> None:
        """Output tensor must be float32.

        Args:
            encoder: FluidBiomarkerEncoder fixture.
        """
        x = _make_fluid_valid()
        with torch.no_grad():
            out = encoder(x)
        assert out.dtype == torch.float32, (
            f"Expected torch.float32, got {out.dtype}"
        )

    def test_fluid_valid_input_range(self, encoder: FluidBiomarkerEncoder) -> None:
        """Forward pass must complete without error for valid mid-range inputs.

        Args:
            encoder: FluidBiomarkerEncoder fixture.
        """
        x = _make_fluid_valid()
        with torch.no_grad():
            out = encoder(x)
        assert out is not None

    def test_fluid_invalid_ptau_raises(self, encoder: FluidBiomarkerEncoder) -> None:
        """pTau-217 above 100 pg/mL must raise ValueError.

        Args:
            encoder: FluidBiomarkerEncoder fixture.
        """
        x = _make_fluid_valid()
        x[:, 0] = 150.0  # pTau-217 above range [0.1, 100]
        with pytest.raises(ValueError, match="pTau-217"):
            encoder(x)

    def test_fluid_invalid_abeta_ratio_low_raises(
        self, encoder: FluidBiomarkerEncoder
    ) -> None:
        """Abeta42/40 ratio below 0.01 must raise ValueError.

        Args:
            encoder: FluidBiomarkerEncoder fixture.
        """
        x = _make_fluid_valid()
        x[:, 1] = 0.001  # below minimum 0.01
        with pytest.raises(ValueError, match="Abeta42/40"):
            encoder(x)

    def test_fluid_invalid_nfl_raises(self, encoder: FluidBiomarkerEncoder) -> None:
        """NfL below 5 pg/mL must raise ValueError.

        Args:
            encoder: FluidBiomarkerEncoder fixture.
        """
        x = _make_fluid_valid()
        x[:, 2] = 1.0  # NfL below range [5, 200]
        with pytest.raises(ValueError, match="NfL"):
            encoder(x)

    def test_fluid_nan_input_raises(self, encoder: FluidBiomarkerEncoder) -> None:
        """NaN in any feature must raise ValueError.

        Args:
            encoder: FluidBiomarkerEncoder fixture.
        """
        x = _make_fluid_valid()
        x[0, 3] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            encoder(x)

    def test_fluid_no_nan_output(self, encoder: FluidBiomarkerEncoder) -> None:
        """Output must contain no NaN values for valid input.

        Args:
            encoder: FluidBiomarkerEncoder fixture.
        """
        x = _make_fluid_valid()
        with torch.no_grad():
            out = encoder(x)
        assert not torch.isnan(out).any(), "Output contains unexpected NaN values"

    def test_fluid_boundary_values(self, encoder: FluidBiomarkerEncoder) -> None:
        """Exact boundary values (min and max) must be accepted without error.

        Args:
            encoder: FluidBiomarkerEncoder fixture.
        """
        x = _make_fluid_valid()
        # Exact lower bounds
        x[:, 0] = 0.1    # pTau-217 min
        x[:, 1] = 0.01   # Abeta42/40 min
        x[:, 2] = 5.0    # NfL min
        with torch.no_grad():
            out_low = encoder(x)
        assert not torch.isnan(out_low).any()

        # Exact upper bounds
        x[:, 0] = 100.0  # pTau-217 max
        x[:, 1] = 0.30   # Abeta42/40 max
        x[:, 2] = 200.0  # NfL max
        with torch.no_grad():
            out_high = encoder(x)
        assert not torch.isnan(out_high).any()


# ===========================================================================
# DigitalAcousticEncoder tests
# ===========================================================================

class TestDigitalAcousticEncoder:
    """Unit tests for DigitalAcousticEncoder (SRS-001 § 4.2.2)."""

    @pytest.fixture
    def encoder(self) -> DigitalAcousticEncoder:
        """Provide a freshly initialised DigitalAcousticEncoder in eval mode.

        Returns:
            DigitalAcousticEncoder instance set to eval mode.
        """
        model = DigitalAcousticEncoder()
        model.eval()
        return model

    def test_acoustic_output_shape(self, encoder: DigitalAcousticEncoder) -> None:
        """Output tensor must have shape [batch_size, 768].

        Args:
            encoder: DigitalAcousticEncoder fixture.
        """
        x = _make_acoustic_valid()
        with torch.no_grad():
            out = encoder(x)
        assert out.shape == (BATCH_SIZE, EXPECTED_OUTPUT_DIM), (
            f"Expected shape ({BATCH_SIZE}, {EXPECTED_OUTPUT_DIM}), got {out.shape}"
        )

    def test_acoustic_output_type(self, encoder: DigitalAcousticEncoder) -> None:
        """Output tensor must be float32.

        Args:
            encoder: DigitalAcousticEncoder fixture.
        """
        x = _make_acoustic_valid()
        with torch.no_grad():
            out = encoder(x)
        assert out.dtype == torch.float32, (
            f"Expected torch.float32, got {out.dtype}"
        )

    def test_acoustic_valid_input_range(self, encoder: DigitalAcousticEncoder) -> None:
        """Forward pass must complete without error for valid mid-range inputs.

        Args:
            encoder: DigitalAcousticEncoder fixture.
        """
        x = _make_acoustic_valid()
        with torch.no_grad():
            out = encoder(x)
        assert out is not None

    def test_acoustic_invalid_jitter_raises(
        self, encoder: DigitalAcousticEncoder
    ) -> None:
        """Jitter above 0.05 must raise ValueError.

        Args:
            encoder: DigitalAcousticEncoder fixture.
        """
        x = _make_acoustic_valid()
        x[:, 0] = 0.1  # jitter above range [0.0001, 0.05]
        with pytest.raises(ValueError, match="jitter"):
            encoder(x)

    def test_acoustic_invalid_shimmer_raises(
        self, encoder: DigitalAcousticEncoder
    ) -> None:
        """Shimmer above 0.30 must raise ValueError.

        Args:
            encoder: DigitalAcousticEncoder fixture.
        """
        x = _make_acoustic_valid()
        x[:, 1] = 0.5  # shimmer above range [0.001, 0.30]
        with pytest.raises(ValueError, match="shimmer"):
            encoder(x)

    def test_acoustic_nan_input_raises(self, encoder: DigitalAcousticEncoder) -> None:
        """NaN in any feature must raise ValueError.

        Args:
            encoder: DigitalAcousticEncoder fixture.
        """
        x = _make_acoustic_valid()
        x[1, 5] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            encoder(x)

    def test_acoustic_no_nan_output(self, encoder: DigitalAcousticEncoder) -> None:
        """Output must contain no NaN values for valid input.

        Args:
            encoder: DigitalAcousticEncoder fixture.
        """
        x = _make_acoustic_valid()
        with torch.no_grad():
            out = encoder(x)
        assert not torch.isnan(out).any(), "Output contains unexpected NaN values"

    def test_acoustic_boundary_jitter(self, encoder: DigitalAcousticEncoder) -> None:
        """Exact boundary jitter values must be accepted.

        Args:
            encoder: DigitalAcousticEncoder fixture.
        """
        x = _make_acoustic_valid()
        x[:, 0] = 0.0001  # jitter min
        with torch.no_grad():
            out = encoder(x)
        assert not torch.isnan(out).any()

        x[:, 0] = 0.05  # jitter max
        with torch.no_grad():
            out = encoder(x)
        assert not torch.isnan(out).any()


# ===========================================================================
# DigitalMotorEncoder tests
# ===========================================================================

class TestDigitalMotorEncoder:
    """Unit tests for DigitalMotorEncoder (SRS-001 § 4.2.3)."""

    @pytest.fixture
    def encoder(self) -> DigitalMotorEncoder:
        """Provide a freshly initialised DigitalMotorEncoder in eval mode.

        Returns:
            DigitalMotorEncoder instance set to eval mode.
        """
        model = DigitalMotorEncoder()
        model.eval()
        return model

    def test_motor_output_shape(self, encoder: DigitalMotorEncoder) -> None:
        """Output tensor must have shape [batch_size, 768].

        Args:
            encoder: DigitalMotorEncoder fixture.
        """
        x = _make_motor_valid()
        with torch.no_grad():
            out = encoder(x)
        assert out.shape == (BATCH_SIZE, EXPECTED_OUTPUT_DIM), (
            f"Expected shape ({BATCH_SIZE}, {EXPECTED_OUTPUT_DIM}), got {out.shape}"
        )

    def test_motor_output_type(self, encoder: DigitalMotorEncoder) -> None:
        """Output tensor must be float32.

        Args:
            encoder: DigitalMotorEncoder fixture.
        """
        x = _make_motor_valid()
        with torch.no_grad():
            out = encoder(x)
        assert out.dtype == torch.float32, (
            f"Expected torch.float32, got {out.dtype}"
        )

    def test_motor_valid_input_range(self, encoder: DigitalMotorEncoder) -> None:
        """Forward pass must complete without error for valid mid-range inputs.

        Args:
            encoder: DigitalMotorEncoder fixture.
        """
        x = _make_motor_valid()
        with torch.no_grad():
            out = encoder(x)
        assert out is not None

    def test_motor_nan_input_raises(self, encoder: DigitalMotorEncoder) -> None:
        """NaN in any motor feature must raise ValueError.

        Args:
            encoder: DigitalMotorEncoder fixture.
        """
        x = _make_motor_valid()
        x[2, 4] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            encoder(x)

    def test_motor_no_nan_output(self, encoder: DigitalMotorEncoder) -> None:
        """Output must contain no NaN values for valid input.

        Args:
            encoder: DigitalMotorEncoder fixture.
        """
        x = _make_motor_valid()
        with torch.no_grad():
            out = encoder(x)
        assert not torch.isnan(out).any(), "Output contains unexpected NaN values"

    def test_motor_arbitrary_finite_values_accepted(
        self, encoder: DigitalMotorEncoder
    ) -> None:
        """Motor encoder must accept arbitrary finite values (no range constraints).

        Args:
            encoder: DigitalMotorEncoder fixture.
        """
        torch.manual_seed(42)
        x = torch.randn(BATCH_SIZE, 8, dtype=torch.float32) * 100
        with torch.no_grad():
            out = encoder(x)
        assert out.shape == (BATCH_SIZE, EXPECTED_OUTPUT_DIM)


# ===========================================================================
# ClinicalDemographicEncoder tests
# ===========================================================================

class TestClinicalDemographicEncoder:
    """Unit tests for ClinicalDemographicEncoder (SRS-001 § 4.2.4)."""

    @pytest.fixture
    def encoder(self) -> ClinicalDemographicEncoder:
        """Provide a freshly initialised ClinicalDemographicEncoder in eval mode.

        Returns:
            ClinicalDemographicEncoder instance set to eval mode.
        """
        model = ClinicalDemographicEncoder()
        model.eval()
        return model

    def test_clinical_output_shape(self, encoder: ClinicalDemographicEncoder) -> None:
        """Output tensor must have shape [batch_size, 768].

        Args:
            encoder: ClinicalDemographicEncoder fixture.
        """
        x = _make_clinical_valid()
        with torch.no_grad():
            out = encoder(x)
        assert out.shape == (BATCH_SIZE, EXPECTED_OUTPUT_DIM), (
            f"Expected shape ({BATCH_SIZE}, {EXPECTED_OUTPUT_DIM}), got {out.shape}"
        )

    def test_clinical_output_type(self, encoder: ClinicalDemographicEncoder) -> None:
        """Output tensor must be float32.

        Args:
            encoder: ClinicalDemographicEncoder fixture.
        """
        x = _make_clinical_valid()
        with torch.no_grad():
            out = encoder(x)
        assert out.dtype == torch.float32, (
            f"Expected torch.float32, got {out.dtype}"
        )

    def test_clinical_valid_input_range(
        self, encoder: ClinicalDemographicEncoder
    ) -> None:
        """Forward pass must complete without error for valid mid-range inputs.

        Args:
            encoder: ClinicalDemographicEncoder fixture.
        """
        x = _make_clinical_valid()
        with torch.no_grad():
            out = encoder(x)
        assert out is not None

    def test_clinical_invalid_mmse_high_raises(
        self, encoder: ClinicalDemographicEncoder
    ) -> None:
        """MMSE above 30 must raise ValueError.

        Args:
            encoder: ClinicalDemographicEncoder fixture.
        """
        x = _make_clinical_valid()
        x[:, 3] = 35.0  # MMSE above valid range [0, 30]
        with pytest.raises(ValueError, match="MMSE"):
            encoder(x)

    def test_clinical_invalid_mmse_negative_raises(
        self, encoder: ClinicalDemographicEncoder
    ) -> None:
        """Negative MMSE must raise ValueError.

        Args:
            encoder: ClinicalDemographicEncoder fixture.
        """
        x = _make_clinical_valid()
        x[:, 3] = -1.0  # MMSE below valid range [0, 30]
        with pytest.raises(ValueError, match="MMSE"):
            encoder(x)

    def test_clinical_nan_input_raises(
        self, encoder: ClinicalDemographicEncoder
    ) -> None:
        """NaN in any feature must raise ValueError.

        Args:
            encoder: ClinicalDemographicEncoder fixture.
        """
        x = _make_clinical_valid()
        x[0, 0] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            encoder(x)

    def test_clinical_no_nan_output(
        self, encoder: ClinicalDemographicEncoder
    ) -> None:
        """Output must contain no NaN values for valid input.

        Args:
            encoder: ClinicalDemographicEncoder fixture.
        """
        x = _make_clinical_valid()
        with torch.no_grad():
            out = encoder(x)
        assert not torch.isnan(out).any(), "Output contains unexpected NaN values"

    def test_clinical_mmse_boundary_values(
        self, encoder: ClinicalDemographicEncoder
    ) -> None:
        """MMSE boundary values 0 and 30 must be accepted.

        Args:
            encoder: ClinicalDemographicEncoder fixture.
        """
        x = _make_clinical_valid()
        x[:, 3] = 0.0  # MMSE minimum
        with torch.no_grad():
            out_low = encoder(x)
        assert not torch.isnan(out_low).any()

        x[:, 3] = 30.0  # MMSE maximum
        with torch.no_grad():
            out_high = encoder(x)
        assert not torch.isnan(out_high).any()


# ===========================================================================
# Cross-encoder consistency tests
# ===========================================================================

class TestEncoderConsistency:
    """Cross-encoder consistency tests ensuring architectural uniformity."""

    def test_all_encoders_share_output_dim(self) -> None:
        """All four encoders must output the same embedding dimension (768).

        This ensures compatibility with the downstream CrossModalAttention
        module which expects all modality embeddings to be equal size.
        """
        assert FluidBiomarkerEncoder.OUTPUT_DIM == EXPECTED_OUTPUT_DIM
        assert DigitalAcousticEncoder.OUTPUT_DIM == EXPECTED_OUTPUT_DIM
        assert DigitalMotorEncoder.OUTPUT_DIM == EXPECTED_OUTPUT_DIM
        assert ClinicalDemographicEncoder.OUTPUT_DIM == EXPECTED_OUTPUT_DIM

    def test_all_encoders_are_nn_module_subclasses(self) -> None:
        """All encoders must subclass torch.nn.Module.

        Required for PyTorch serialisation, device transfer, and
        gradient tracking compatibility.
        """
        for cls in (
            FluidBiomarkerEncoder,
            DigitalAcousticEncoder,
            DigitalMotorEncoder,
            ClinicalDemographicEncoder,
        ):
            assert issubclass(cls, nn.Module), (
                f"{cls.__name__} does not subclass torch.nn.Module"
            )

    def test_all_encoders_eval_train_toggle(self) -> None:
        """All encoders must correctly toggle between train and eval modes.

        This verifies Dropout behaves differently in train vs. eval, which
        is critical for reproducible inference.
        """
        for cls in (
            FluidBiomarkerEncoder,
            DigitalAcousticEncoder,
            DigitalMotorEncoder,
            ClinicalDemographicEncoder,
        ):
            model = cls()
            model.train()
            assert model.training, (
                f"{cls.__name__} did not enter train mode"
            )
            model.eval()
            assert not model.training, (
                f"{cls.__name__} did not enter eval mode"
            )

    def test_all_encoders_deterministic_in_eval(self) -> None:
        """All encoders must produce identical outputs on consecutive passes in eval.

        Dropout must be disabled in eval mode, ensuring reproducible inference.
        """
        test_cases = [
            (FluidBiomarkerEncoder(), _make_fluid_valid()),
            (DigitalAcousticEncoder(), _make_acoustic_valid()),
            (DigitalMotorEncoder(), _make_motor_valid()),
            (ClinicalDemographicEncoder(), _make_clinical_valid()),
        ]
        for model, x in test_cases:
            model.eval()
            with torch.no_grad():
                out1 = model(x)
                out2 = model(x)
            assert torch.allclose(out1, out2), (
                f"{model.__class__.__name__} produced non-deterministic output "
                "in eval mode"
            )
