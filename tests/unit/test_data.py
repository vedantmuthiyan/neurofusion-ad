"""Unit tests for NeuroFusion-AD data pipeline components.

Tests cover InputValidator, ADNIPreprocessor, DigitalBiomarkerSynthesizer,
NeuroFusionDataset, and DataLoader utilities.

All tests use only synthetic data — no real patient data is required.

IEC 62304 Requirement Traceability: SRS-001 § 5.x (Data Pipeline)
"""

import math
import pytest
import torch

from src.data.validators import InputValidator
from src.data.adni_preprocessing import ADNIPreprocessor
from src.data.digital_biomarker_synthesis import DigitalBiomarkerSynthesizer
from src.data.dataset import (
    NeuroFusionDataset,
    generate_synthetic_adni,
    create_dataloaders,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthesizer() -> DigitalBiomarkerSynthesizer:
    """Return a DigitalBiomarkerSynthesizer with a fixed seed."""
    return DigitalBiomarkerSynthesizer(seed=0)


@pytest.fixture
def validator() -> InputValidator:
    """Return an InputValidator instance."""
    return InputValidator()


@pytest.fixture
def preprocessor() -> ADNIPreprocessor:
    """Return an ADNIPreprocessor with default mean-imputation strategy."""
    return ADNIPreprocessor(missing_strategy="mean")


@pytest.fixture
def valid_fluid() -> torch.Tensor:
    """Return a valid fluid biomarker tensor of shape [6]."""
    # ptau217=10.0, abeta42_40=0.10, nfl=30.0, gfap=150, total_tau=250, abeta42=800
    return torch.tensor([10.0, 0.10, 30.0, 150.0, 250.0, 800.0])


@pytest.fixture
def valid_acoustic() -> torch.Tensor:
    """Return a valid acoustic feature tensor of shape [12]."""
    return torch.tensor([0.005, 0.04, 15.0, 130.0, 25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.fixture
def valid_motor() -> torch.Tensor:
    """Return a valid motor feature tensor of shape [8]."""
    return torch.tensor([4.0, 0.3, 50.0, 2.0, 0.15, 0.05, 25.0, 0.10])


@pytest.fixture
def valid_clinical() -> torch.Tensor:
    """Return a valid clinical feature tensor of shape [10]."""
    # age=72, edu=14, sex=0, mmse=26, cdr=1.5, gds=5, bmi=27, sbp=130, apoe4=0, comorbidities=2
    return torch.tensor([72.0, 14.0, 0.0, 26.0, 1.5, 5.0, 27.0, 130.0, 0.0, 2.0])


# ---------------------------------------------------------------------------
# InputValidator tests
# ---------------------------------------------------------------------------


class TestInputValidator:
    """Tests for InputValidator clinical range enforcement."""

    def test_validator_fluid_valid(
        self, validator: InputValidator, valid_fluid: torch.Tensor
    ) -> None:
        """Test 1: Valid fluid biomarker tensor passes without error."""
        # Should not raise
        validator.validate_fluid_biomarkers(valid_fluid)

    def test_validator_fluid_ptau_too_high(self, validator: InputValidator) -> None:
        """Test 2: ptau217 > 100 pg/mL raises ValueError."""
        # ptau217 at index 0, above max of 100.0
        invalid = torch.tensor([101.0, 0.10, 30.0, 150.0, 250.0, 800.0])
        with pytest.raises(ValueError, match="ptau217"):
            validator.validate_fluid_biomarkers(invalid)

    def test_validator_fluid_abeta_ratio_too_low(self, validator: InputValidator) -> None:
        """Test 3: abeta42_40_ratio < 0.01 raises ValueError."""
        # abeta42_40_ratio at index 1, below min of 0.01
        invalid = torch.tensor([10.0, 0.005, 30.0, 150.0, 250.0, 800.0])
        with pytest.raises(ValueError, match="abeta42_40_ratio"):
            validator.validate_fluid_biomarkers(invalid)

    def test_validator_fluid_nfl_too_high(self, validator: InputValidator) -> None:
        """Test 4: NfL > 200 pg/mL raises ValueError."""
        # nfl at index 2, above max of 200.0
        invalid = torch.tensor([10.0, 0.10, 201.0, 150.0, 250.0, 800.0])
        with pytest.raises(ValueError, match="nfl"):
            validator.validate_fluid_biomarkers(invalid)

    def test_validator_acoustic_jitter_invalid(self, validator: InputValidator) -> None:
        """Test 5: Acoustic jitter > 0.05 raises ValueError."""
        # jitter at index 0, above max of 0.05
        invalid = torch.tensor([0.06, 0.04, 15.0, 130.0, 25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="jitter"):
            validator.validate_acoustic_features(invalid)

    def test_validator_clinical_mmse_negative(self, validator: InputValidator) -> None:
        """Test 6: MMSE < 0 raises ValueError."""
        # mmse at index 3, below min of 0
        invalid = torch.tensor([72.0, 14.0, 0.0, -1.0, 1.5, 5.0, 27.0, 130.0, 0.0, 2.0])
        with pytest.raises(ValueError, match="mmse"):
            validator.validate_clinical_features(invalid)

    def test_validator_clinical_mmse_over_30(self, validator: InputValidator) -> None:
        """Test 7: MMSE > 30 raises ValueError."""
        # mmse at index 3, above max of 30
        invalid = torch.tensor([72.0, 14.0, 0.0, 31.0, 1.5, 5.0, 27.0, 130.0, 0.0, 2.0])
        with pytest.raises(ValueError, match="mmse"):
            validator.validate_clinical_features(invalid)


# ---------------------------------------------------------------------------
# ADNIPreprocessor tests
# ---------------------------------------------------------------------------


class TestADNIPreprocessor:
    """Tests for ADNIPreprocessor normalization and imputation."""

    def test_preprocessor_normalize_shape(self, preprocessor: ADNIPreprocessor) -> None:
        """Test 8: normalize returns a tensor of the same shape as the input.

        Phase 2B: fluid is [ptau217, nfl_plasma] — 2 features.
        FLUID_MEAN and FLUID_STD are now 2-element tensors.
        """
        features = torch.tensor([10.0, 30.0])  # Phase 2B: [ptau217, nfl_plasma]
        result = preprocessor.normalize(features, ADNIPreprocessor.FLUID_MEAN, ADNIPreprocessor.FLUID_STD)
        assert result.shape == features.shape, (
            f"normalize changed shape from {features.shape} to {result.shape}"
        )

    def test_preprocessor_impute_nan(self, preprocessor: ADNIPreprocessor) -> None:
        """Test 9: impute_missing replaces all NaN values.

        Phase 2B: fluid is 2 features — uses 2-element FLUID_MEAN.
        """
        features = torch.tensor([float("nan"), float("nan")])  # Phase 2B: [ptau217, nfl_plasma]
        result = preprocessor.impute_missing(features, ADNIPreprocessor.FLUID_MEAN)
        assert not torch.isnan(result).any(), "impute_missing left NaN values in the tensor."
        assert result.shape == features.shape

    def test_preprocessor_preprocess_record(self, preprocessor: ADNIPreprocessor) -> None:
        """Test 10: preprocess_record returns dict with correct keys and tensor shapes.

        Phase 2B: fluid is [ptau217, nfl_plasma] — 2 features.
        """
        record = {
            "fluid":    [10.0, 30.0],  # Phase 2B: [ptau217, nfl_plasma]
            "acoustic": [0.005, 0.04, 15.0, 130.0, 25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "motor":    [4.0, 0.3, 50.0, 2.0, 0.15, 0.05, 25.0, 0.10],
            "clinical": [72.0, 14.0, 0.0, 26.0, 1.5, 5.0, 27.0, 130.0, 0.0, 2.0],
        }
        result = preprocessor.preprocess_record(record)

        assert set(result.keys()) == {"fluid", "acoustic", "motor", "clinical"}
        assert result["fluid"].shape == (2,), f"fluid shape: {result['fluid'].shape}"  # Phase 2B
        assert result["acoustic"].shape == (12,), f"acoustic shape: {result['acoustic'].shape}"
        assert result["motor"].shape == (8,), f"motor shape: {result['motor'].shape}"
        assert result["clinical"].shape == (10,), f"clinical shape: {result['clinical'].shape}"

        # Verify normalization was applied (values should be in [-5, 5])
        for key, tensor in result.items():
            assert tensor.min() >= -5.0 and tensor.max() <= 5.0, (
                f"Normalized '{key}' tensor has values outside [-5, 5]."
            )


# ---------------------------------------------------------------------------
# DigitalBiomarkerSynthesizer tests
# ---------------------------------------------------------------------------


class TestDigitalBiomarkerSynthesizer:
    """Tests for DigitalBiomarkerSynthesizer output shapes and clinical ranges."""

    def test_synthesizer_acoustic_shape(self, synthesizer: DigitalBiomarkerSynthesizer) -> None:
        """Test 11: synthesize_acoustic returns tensor of shape [n_samples, 12]."""
        n = 50
        result = synthesizer.synthesize_acoustic(n)
        assert result.shape == (n, 12), f"Expected ({n}, 12), got {result.shape}"

    def test_synthesizer_motor_shape(self, synthesizer: DigitalBiomarkerSynthesizer) -> None:
        """Test 12: synthesize_motor returns tensor of shape [n_samples, 8]."""
        n = 50
        result = synthesizer.synthesize_motor(n)
        assert result.shape == (n, 8), f"Expected ({n}, 8), got {result.shape}"

    def test_synthesizer_fluid_shape(self, synthesizer: DigitalBiomarkerSynthesizer) -> None:
        """Test 13: synthesize_fluid_biomarkers returns tensor of shape [n_samples, 6]."""
        n = 50
        result = synthesizer.synthesize_fluid_biomarkers(n)
        assert result.shape == (n, 6), f"Expected ({n}, 6), got {result.shape}"

    def test_synthesizer_fluid_ranges(self, synthesizer: DigitalBiomarkerSynthesizer) -> None:
        """Test 14: All fluid biomarker validated features are within clinical ranges."""
        n = 500
        result = synthesizer.synthesize_fluid_biomarkers(n)

        # ptau217 (index 0): [0.1, 100.0]
        assert result[:, 0].min() >= 0.1, f"ptau217 below 0.1: {result[:, 0].min()}"
        assert result[:, 0].max() <= 100.0, f"ptau217 above 100.0: {result[:, 0].max()}"

        # abeta42_40_ratio (index 1): [0.01, 0.30]
        assert result[:, 1].min() >= 0.01, f"abeta42_40_ratio below 0.01: {result[:, 1].min()}"
        assert result[:, 1].max() <= 0.30, f"abeta42_40_ratio above 0.30: {result[:, 1].max()}"

        # nfl (index 2): [5.0, 200.0]
        assert result[:, 2].min() >= 5.0, f"nfl below 5.0: {result[:, 2].min()}"
        assert result[:, 2].max() <= 200.0, f"nfl above 200.0: {result[:, 2].max()}"

    def test_synthesizer_acoustic_jitter_range(
        self, synthesizer: DigitalBiomarkerSynthesizer
    ) -> None:
        """Test 15: Synthesized jitter (acoustic index 0) is within [0.0001, 0.05]."""
        n = 500
        result = synthesizer.synthesize_acoustic(n)
        jitter = result[:, 0]
        assert jitter.min() >= 0.0001, f"jitter below 0.0001: {jitter.min()}"
        assert jitter.max() <= 0.05, f"jitter above 0.05: {jitter.max()}"

    def test_synthesizer_clinical_mmse_range(
        self, synthesizer: DigitalBiomarkerSynthesizer
    ) -> None:
        """Test 16: Synthesized MMSE (clinical index 3) is within [0, 30]."""
        n = 500
        result = synthesizer.synthesize_clinical(n)
        mmse = result[:, 3]
        assert mmse.min() >= 0.0, f"MMSE below 0: {mmse.min()}"
        assert mmse.max() <= 30.0, f"MMSE above 30: {mmse.max()}"

    def test_synthesizer_full_dataset_keys(
        self, synthesizer: DigitalBiomarkerSynthesizer
    ) -> None:
        """Test 17: synthesize_full_dataset returns all required keys."""
        required_keys = {
            "fluid", "acoustic", "motor", "clinical",
            "amyloid_label", "mmse_slope", "time_to_event", "event_observed",
        }
        result = synthesizer.synthesize_full_dataset(n_samples=20)
        assert required_keys == set(result.keys()), (
            f"Missing keys: {required_keys - set(result.keys())}"
        )


# ---------------------------------------------------------------------------
# NeuroFusionDataset tests
# ---------------------------------------------------------------------------


def _make_dataset(n: int = 50) -> NeuroFusionDataset:
    """Create a small synthetic NeuroFusionDataset for testing."""
    synth = DigitalBiomarkerSynthesizer(seed=1)
    data = synth.synthesize_full_dataset(n_samples=n)
    return NeuroFusionDataset(data=data)


class TestNeuroFusionDataset:
    """Tests for NeuroFusionDataset length, item access, and shapes."""

    def test_dataset_len(self) -> None:
        """Test 18: Dataset __len__ returns the correct number of samples."""
        n = 80
        ds = _make_dataset(n)
        assert len(ds) == n, f"Expected {n}, got {len(ds)}"

    def test_dataset_getitem_keys(self) -> None:
        """Test 19: __getitem__ returns a dict with all required keys."""
        ds = _make_dataset(30)
        item = ds[0]
        required_keys = {
            "fluid", "acoustic", "motor", "clinical",
            "amyloid_label", "mmse_slope", "time_to_event", "event_observed",
        }
        assert required_keys == set(item.keys()), (
            f"Missing keys: {required_keys - set(item.keys())}"
        )

    def test_dataset_getitem_shapes(self) -> None:
        """Test 20: __getitem__ returns tensors with correct per-sample shapes."""
        ds = _make_dataset(30)
        item = ds[0]

        assert item["fluid"].shape == (6,), f"fluid: {item['fluid'].shape}"
        assert item["acoustic"].shape == (12,), f"acoustic: {item['acoustic'].shape}"
        assert item["motor"].shape == (8,), f"motor: {item['motor'].shape}"
        assert item["clinical"].shape == (10,), f"clinical: {item['clinical'].shape}"

        # Scalar labels
        for label_key in ("amyloid_label", "mmse_slope", "time_to_event", "event_observed"):
            assert item[label_key].shape == (), (
                f"Label '{label_key}' is not scalar: {item[label_key].shape}"
            )


# ---------------------------------------------------------------------------
# create_dataloaders tests
# ---------------------------------------------------------------------------


class TestCreateDataloaders:
    """Tests for create_dataloaders split sizes and batch shapes."""

    def test_dataloaders_sizes(self) -> None:
        """Test 21: train+val+test sizes add up to total dataset size."""
        n = 100
        ds = _make_dataset(n)
        train_loader, val_loader, test_loader = create_dataloaders(
            ds, train_fraction=0.7, val_fraction=0.15, batch_size=16, seed=42
        )

        n_train = len(train_loader.dataset)  # type: ignore[arg-type]
        n_val = len(val_loader.dataset)      # type: ignore[arg-type]
        n_test = len(test_loader.dataset)    # type: ignore[arg-type]

        assert n_train + n_val + n_test == n, (
            f"Split sizes ({n_train}+{n_val}+{n_test}={n_train+n_val+n_test}) "
            f"do not add up to {n}."
        )
        assert n_train == 70, f"Expected 70 train samples, got {n_train}"
        assert n_val == 15, f"Expected 15 val samples, got {n_val}"
        assert n_test == 15, f"Expected 15 test samples, got {n_test}"

    def test_dataloaders_batch_shape(self) -> None:
        """Test 22: DataLoader batches have correct shape [batch, feature_dim]."""
        n = 64
        batch_size = 16
        ds = _make_dataset(n)
        train_loader, _, _ = create_dataloaders(ds, batch_size=batch_size, seed=0)

        batch = next(iter(train_loader))
        assert batch["fluid"].shape == (batch_size, 6), (
            f"fluid batch shape: {batch['fluid'].shape}"
        )
        assert batch["acoustic"].shape == (batch_size, 12), (
            f"acoustic batch shape: {batch['acoustic'].shape}"
        )
        assert batch["motor"].shape == (batch_size, 8), (
            f"motor batch shape: {batch['motor'].shape}"
        )
        assert batch["clinical"].shape == (batch_size, 10), (
            f"clinical batch shape: {batch['clinical'].shape}"
        )
        assert batch["amyloid_label"].shape == (batch_size,), (
            f"amyloid_label batch shape: {batch['amyloid_label'].shape}"
        )

    def test_generate_synthetic_adni(self) -> None:
        """Test 23: generate_synthetic_adni completes without errors and returns a NeuroFusionDataset."""
        ds = generate_synthetic_adni(n_samples=100, seed=7)
        assert isinstance(ds, NeuroFusionDataset), (
            f"Expected NeuroFusionDataset, got {type(ds)}"
        )
        assert len(ds) == 100, f"Expected 100 samples, got {len(ds)}"
        # Verify a sample can be retrieved
        item = ds[0]
        assert "fluid" in item
        assert "amyloid_label" in item
