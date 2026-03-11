"""Unit tests for NeuroFusion-AD training infrastructure.

Covers:
    - MultiTaskLoss (losses.py): masked loss computation, NaN handling, weight=0 skipping
    - cox_partial_likelihood_loss: valid inputs, no-event edge case
    - augment_batch: noise addition, label preservation
    - NeuroFusionCSVDataset (csv_dataset.py): ADNI + BH column mapping, NaN imputation
    - NeuroFusionTrainer (trainer.py): train_epoch, early stopping

All tests use only synthetic data — no real patient data required.
Tests use torch.zeros/ones/randn for tensor creation (avoids numpy 2.x ABI issues).

IEC 62304 Requirement Traceability:
    SRS-001 § 5.5 — Training Requirements (tests verify training correctness)
    SRS-001 § 5.4 — Data Loading (tests verify dataset shapes + imputation)
"""

from __future__ import annotations

import csv
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from src.training.losses import MultiTaskLoss, cox_partial_likelihood_loss, augment_batch


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_predictions(batch_size: int = 8, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Create a dummy predictions dict matching NeuroFusionAD.forward() output.

    Args:
        batch_size: Number of samples.
        device: Target device.

    Returns:
        Dict with 'amyloid_logit', 'mmse_slope', 'cox_log_hazard'.
    """
    return {
        "amyloid_logit": torch.randn(batch_size, 1, device=device),
        "mmse_slope": torch.randn(batch_size, 1, device=device),
        "cox_log_hazard": torch.randn(batch_size, 1, device=device),
    }


def _make_targets_full(batch_size: int = 8, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Create a dummy targets dict with fully observed (no NaN) labels.

    Args:
        batch_size: Number of samples.
        device: Target device.

    Returns:
        Dict with 'amyloid_label', 'mmse_slope', 'survival_time', 'event_indicator'.
    """
    return {
        "amyloid_label": torch.randint(0, 2, (batch_size,), device=device).float(),
        "mmse_slope": torch.randn(batch_size, device=device),
        "survival_time": torch.rand(batch_size, device=device) * 24.0 + 1.0,
        "event_indicator": torch.randint(0, 2, (batch_size,), device=device).float(),
    }


def _make_targets_nan_cls(batch_size: int = 8, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Create targets with all-NaN amyloid labels.

    Args:
        batch_size: Number of samples.
        device: Target device.

    Returns:
        Targets dict where 'amyloid_label' is entirely NaN.
    """
    targets = _make_targets_full(batch_size, device)
    targets["amyloid_label"] = torch.full((batch_size,), float("nan"), device=device)
    return targets


def _make_minimal_model() -> nn.Module:
    """Create a minimal nn.Module that mimics NeuroFusionAD.forward() output.

    The full NeuroFusionAD requires torch-geometric which may not be available
    in all test environments. This minimal model is sufficient for trainer tests.

    Returns:
        Small linear model with compatible forward() signature.
    """
    class _MinimalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc_fluid = nn.Linear(6, 16)
            self.fc_acoustic = nn.Linear(12, 16)
            self.fc_motor = nn.Linear(8, 16)
            self.fc_clinical = nn.Linear(10, 16)
            self.amyloid_head = nn.Linear(64, 1)
            self.mmse_head = nn.Linear(64, 1)
            self.surv_head = nn.Linear(64, 1)

        def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            f = torch.relu(self.fc_fluid(batch["fluid"]))
            a = torch.relu(self.fc_acoustic(batch["acoustic"]))
            m = torch.relu(self.fc_motor(batch["motor"]))
            c = torch.relu(self.fc_clinical(batch["clinical"]))
            fused = torch.cat([f, a, m, c], dim=-1)
            return {
                "amyloid_logit": self.amyloid_head(fused),
                "mmse_slope": self.mmse_head(fused),
                "cox_log_hazard": self.surv_head(fused),
                "fused_embedding": fused,
                "disclaimer": "This tool is intended to support, not replace, clinical judgment.",
            }

    return _MinimalModel()


def _make_batch(batch_size: int = 4, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Create a batch dict for the minimal model.

    Args:
        batch_size: Number of samples.
        device: Target device.

    Returns:
        Dict with 'fluid', 'acoustic', 'motor', 'clinical' tensors.
    """
    return {
        "fluid": torch.randn(batch_size, 6, device=device),
        "acoustic": torch.randn(batch_size, 12, device=device),
        "motor": torch.randn(batch_size, 8, device=device),
        "clinical": torch.randn(batch_size, 10, device=device),
        "amyloid_label": torch.randint(0, 2, (batch_size,), device=device).float(),
        "mmse_slope": torch.randn(batch_size, device=device),
        "survival_time": torch.rand(batch_size, device=device) * 24.0 + 1.0,
        "event_indicator": torch.randint(0, 2, (batch_size,), device=device).float(),
    }


def _write_adni_csv(path: str, n_rows: int = 10, include_nans: bool = False) -> None:
    """Write a minimal ADNI-format CSV for dataset testing.

    Args:
        path: Destination file path.
        n_rows: Number of data rows.
        include_nans: If True, include NaN values in nullable columns.
    """
    cols = [
        "RID",
        "PTAU217", "ABETA4240_RATIO", "NFL_PLASMA", "GFAP_PLASMA", "PTAU181_CSF", "ABETA42_CSF",
        "acoustic_jitter", "acoustic_shimmer", "acoustic_hnr", "acoustic_f0_mean", "acoustic_f0_std",
        "acoustic_mfcc1", "acoustic_mfcc2", "acoustic_mfcc3", "acoustic_mfcc4",
        "acoustic_mfcc5", "acoustic_mfcc6", "acoustic_mfcc7",
        "motor_tremor_freq", "motor_tremor_amp", "motor_bradykinesia_score", "motor_spiral_rmse",
        "motor_tapping_cv", "motor_tapping_asymmetry", "motor_grip_force_mean", "motor_grip_force_cv",
        "AGE", "SEX_CODE", "EDUCATION_YEARS", "MMSE_BASELINE",
        "APOE4_COUNT", "TAU_CSF", "ABETA42_PLASMA", "ABETA40_PLASMA",
        "AMYLOID_POSITIVE", "MMSE_SLOPE", "TIME_TO_EVENT", "EVENT_INDICATOR",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for i in range(n_rows):
            row = {c: str(float(i + 1)) for c in cols}
            row["RID"] = str(i + 1)
            row["ABETA4240_RATIO"] = "0.12"
            row["AMYLOID_POSITIVE"] = "1.0" if i % 2 == 0 else "0.0"
            row["EVENT_INDICATOR"] = "1.0" if i % 3 == 0 else "0.0"
            if include_nans:
                if i % 3 == 0:
                    row["TAU_CSF"] = ""          # 33% null
                if i % 5 == 0:
                    row["AMYLOID_POSITIVE"] = "" # 20% null
                    row["MMSE_SLOPE"] = ""       # 20% null
            writer.writerow(row)


def _write_biohermes_csv(path: str, n_rows: int = 10) -> None:
    """Write a minimal Bio-Hermes-001-format CSV for dataset testing.

    Args:
        path: Destination file path.
        n_rows: Number of data rows.
    """
    cols = [
        "USUBJID",
        "PTAU217", "ABETA4240_RATIO", "NFL_PLASMA", "GFAP_PLASMA", "PTAU181_PLASMA", "ABETA42_PLASMA",
        "acoustic_delayed_recall", "acoustic_object_recall", "acoustic_image_descr_score",
        "acoustic_intraword_pause", "acoustic_speaking_rate", "acoustic_verbal_fluency",
        "acoustic_image_speaking_rate", "acoustic_naming_duration",
        "acoustic_monotonicity", "acoustic_pause_rate",
        "motor_dcr_clock_score", "motor_dcr_delayed_recall", "motor_dcr_score",
        "motor_sdmt_acc", "motor_sdmt_attempted", "motor_spiral_cw_dom",
        "motor_trails_b_acc", "motor_trails_b_time",
        "AGE", "SEX_CODE", "EDUCATION_YEARS", "MMSE_BASELINE",
        "AMYLOID_POSITIVE",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for i in range(n_rows):
            row = {c: str(float(i + 1)) for c in cols}
            row["USUBJID"] = f"BIO-HERMES-00101-{i+1:03d}"
            row["ABETA4240_RATIO"] = "0.12"
            row["AMYLOID_POSITIVE"] = "1.0" if i % 2 == 0 else "0.0"
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Test 1: MultiTaskLoss with fully-observed labels returns finite loss
# ---------------------------------------------------------------------------


class TestMultiTaskLossFullyObserved:
    """Test MultiTaskLoss with fully-observed (non-NaN) labels."""

    def test_fully_observed_returns_finite_total(self) -> None:
        """Test 1: MultiTaskLoss with fully-observed labels returns finite total loss."""
        loss_fn = MultiTaskLoss({"cls": 1.0, "reg": 0.5, "surv": 0.5})
        preds = _make_predictions(batch_size=16)
        targets = _make_targets_full(batch_size=16)
        result = loss_fn(preds, targets)

        assert "total" in result, "Result must have 'total' key"
        assert torch.isfinite(result["total"]), (
            f"Total loss is not finite: {result['total']}"
        )
        assert result["total"].item() > 0.0, "Total loss should be positive"

    def test_fully_observed_all_components_present(self) -> None:
        """Test 2: MultiTaskLoss result dict has all required component keys."""
        loss_fn = MultiTaskLoss()
        preds = _make_predictions(8)
        targets = _make_targets_full(8)
        result = loss_fn(preds, targets)

        for key in ("total", "cls", "reg", "surv"):
            assert key in result, f"Missing key '{key}' in loss result"
            assert isinstance(result[key], torch.Tensor), f"'{key}' must be a tensor"

    def test_fully_observed_cls_loss_finite(self) -> None:
        """Test 3: Classification loss component is finite with valid binary labels."""
        loss_fn = MultiTaskLoss({"cls": 1.0, "reg": 0.0, "surv": 0.0})
        preds = _make_predictions(8)
        targets = _make_targets_full(8)
        result = loss_fn(preds, targets)

        assert torch.isfinite(result["cls"]), f"cls loss not finite: {result['cls']}"
        assert result["cls"].item() > 0.0, "cls loss should be positive for random predictions"


# ---------------------------------------------------------------------------
# Test 4: MultiTaskLoss with all-NaN classification labels skips cls
# ---------------------------------------------------------------------------


class TestMultiTaskLossNaNHandling:
    """Test MultiTaskLoss NaN masking behavior."""

    def test_all_nan_amyloid_labels_skips_cls(self) -> None:
        """Test 4: All-NaN amyloid_labels causes cls component to be zero (skipped)."""
        loss_fn = MultiTaskLoss({"cls": 1.0, "reg": 0.0, "surv": 0.0})
        preds = _make_predictions(8)
        targets = _make_targets_nan_cls(8)
        result = loss_fn(preds, targets)

        # When all cls labels are NaN, cls loss must be zero (not computed)
        assert result["cls"].item() == pytest.approx(0.0, abs=1e-6), (
            f"Expected cls_loss=0 when all labels are NaN, got {result['cls'].item()}"
        )

    def test_partial_nan_amyloid_uses_valid_subset(self) -> None:
        """Test 5: Partial NaN amyloid labels: cls is computed only on non-NaN subset."""
        loss_fn = MultiTaskLoss({"cls": 1.0, "reg": 0.0, "surv": 0.0})
        preds = _make_predictions(8)
        targets = _make_targets_full(8)
        # Set half the amyloid labels to NaN
        targets["amyloid_label"] = targets["amyloid_label"].clone()
        targets["amyloid_label"][:4] = float("nan")

        result_partial = loss_fn(preds, targets)
        result_full_nan = loss_fn(preds, _make_targets_nan_cls(8))

        # Partial NaN should still produce nonzero cls loss
        assert result_partial["cls"].item() > 0.0, (
            "cls loss should be > 0 when some labels are non-NaN"
        )
        # All NaN should produce zero cls loss
        assert result_full_nan["cls"].item() == pytest.approx(0.0, abs=1e-6)

    def test_nan_reg_labels_skips_reg(self) -> None:
        """Test 6: All-NaN MMSE slope labels causes reg component to be zero."""
        loss_fn = MultiTaskLoss({"cls": 0.0, "reg": 1.0, "surv": 0.0})
        preds = _make_predictions(8)
        targets = _make_targets_full(8)
        targets["mmse_slope"] = torch.full((8,), float("nan"))
        result = loss_fn(preds, targets)

        assert result["reg"].item() == pytest.approx(0.0, abs=1e-6), (
            f"Expected reg_loss=0 when all MMSE slopes are NaN, got {result['reg'].item()}"
        )


# ---------------------------------------------------------------------------
# Test 7: Bio-Hermes mode (reg=0, surv=0) only returns cls loss
# ---------------------------------------------------------------------------


class TestMultiTaskLossBioHermesMode:
    """Test MultiTaskLoss with Bio-Hermes-001 weight configuration."""

    def test_biohermes_mode_only_cls_active(self) -> None:
        """Test 7: With reg_weight=0, surv_weight=0, only cls contributes to total."""
        loss_fn = MultiTaskLoss({"cls": 1.0, "reg": 0.0, "surv": 0.0})
        preds = _make_predictions(8)
        targets = _make_targets_full(8)
        result = loss_fn(preds, targets)

        # reg and surv must be exactly zero (weight is 0)
        assert result["reg"].item() == pytest.approx(0.0, abs=1e-6), (
            f"reg should be 0.0 when weight=0, got {result['reg'].item()}"
        )
        assert result["surv"].item() == pytest.approx(0.0, abs=1e-6), (
            f"surv should be 0.0 when weight=0, got {result['surv'].item()}"
        )
        # total == cls (since reg=0, surv=0)
        assert result["total"].item() == pytest.approx(result["cls"].item(), abs=1e-5), (
            "In BH mode, total should equal cls"
        )
        # cls must be positive (real computation happened)
        assert result["cls"].item() > 0.0


# ---------------------------------------------------------------------------
# Test 8 & 9: Cox partial likelihood loss
# ---------------------------------------------------------------------------


class TestCoxPartialLikelihoodLoss:
    """Tests for cox_partial_likelihood_loss."""

    def test_cox_loss_finite_with_valid_inputs(self) -> None:
        """Test 8: Cox loss returns a finite scalar with valid inputs (events present)."""
        torch.manual_seed(42)
        n = 16
        log_hazard = torch.randn(n, 1)
        time = torch.rand(n) * 24.0 + 1.0
        event = torch.zeros(n)
        event[:8] = 1.0  # Ensure at least 8 events

        loss = cox_partial_likelihood_loss(log_hazard, time, event)
        assert torch.isfinite(loss), f"Cox loss not finite: {loss}"
        assert loss.item() >= 0.0, "Cox loss should be non-negative"

    def test_cox_loss_zero_when_no_events(self) -> None:
        """Test 9: Cox loss returns zero tensor when no events in batch."""
        n = 8
        log_hazard = torch.randn(n, 1)
        time = torch.rand(n) * 24.0 + 1.0
        event = torch.zeros(n)  # All censored, no events

        loss = cox_partial_likelihood_loss(log_hazard, time, event)
        assert loss.item() == pytest.approx(0.0, abs=1e-8), (
            f"Cox loss should be 0.0 when no events present, got {loss.item()}"
        )

    def test_cox_loss_zero_when_all_nan_time(self) -> None:
        """Test 10: Cox loss returns zero tensor when all times are NaN."""
        n = 8
        log_hazard = torch.randn(n, 1)
        time = torch.full((n,), float("nan"))
        event = torch.ones(n)

        loss = cox_partial_likelihood_loss(log_hazard, time, event)
        assert loss.item() == pytest.approx(0.0, abs=1e-8), (
            f"Cox loss should be 0.0 when all times are NaN, got {loss.item()}"
        )

    def test_cox_loss_ignores_nan_time_rows(self) -> None:
        """Test 11: Cox loss skips NaN-time rows and computes on valid subset."""
        torch.manual_seed(7)
        n = 16
        log_hazard = torch.randn(n, 1)
        time = torch.rand(n) * 24.0 + 1.0
        event = torch.zeros(n)
        event[:6] = 1.0

        # First 4 rows have NaN time
        time_with_nan = time.clone()
        time_with_nan[:4] = float("nan")

        loss_with_nan = cox_partial_likelihood_loss(log_hazard, time_with_nan, event)
        assert torch.isfinite(loss_with_nan), "Cox loss not finite when subset has NaN times"


# ---------------------------------------------------------------------------
# Test 12 & 13: augment_batch
# ---------------------------------------------------------------------------


class TestAugmentBatch:
    """Tests for augment_batch function."""

    def test_augment_batch_adds_noise(self) -> None:
        """Test 12: augment_batch returns feature tensors different from input."""
        batch = {
            "fluid": torch.zeros(8, 6),
            "acoustic": torch.zeros(8, 12),
            "motor": torch.zeros(8, 8),
            "clinical": torch.zeros(8, 10),
            "amyloid_label": torch.ones(8),
        }
        augmented = augment_batch(batch, noise_std=0.1)

        # All-zero inputs + noise should not be zero
        for key in ("fluid", "acoustic", "motor", "clinical"):
            assert not torch.allclose(augmented[key], batch[key]), (
                f"augment_batch did not modify '{key}' tensor"
            )

    def test_augment_batch_does_not_modify_labels(self) -> None:
        """Test 13: augment_batch does not modify label tensors."""
        label_val = torch.ones(8) * 0.5
        batch = {
            "fluid": torch.randn(8, 6),
            "acoustic": torch.randn(8, 12),
            "motor": torch.randn(8, 8),
            "clinical": torch.randn(8, 10),
            "amyloid_label": label_val.clone(),
            "mmse_slope": torch.randn(8),
            "survival_time": torch.rand(8) * 24,
            "event_indicator": torch.zeros(8),
            "patient_id": "hash_placeholder",
        }
        augmented = augment_batch(batch, noise_std=1.0)

        # Label tensor must be unchanged
        assert torch.allclose(augmented["amyloid_label"], label_val), (
            "augment_batch modified amyloid_label tensor"
        )
        assert augmented["patient_id"] == "hash_placeholder", (
            "augment_batch modified patient_id"
        )

    def test_augment_batch_noise_std_zero_no_change(self) -> None:
        """Test 14: augment_batch with noise_std=0.0 does not modify features."""
        batch = {
            "fluid": torch.ones(4, 6),
            "acoustic": torch.ones(4, 12),
            "motor": torch.ones(4, 8),
            "clinical": torch.ones(4, 10),
        }
        augmented = augment_batch(batch, noise_std=0.0)

        for key in ("fluid", "acoustic", "motor", "clinical"):
            assert torch.allclose(augmented[key], batch[key]), (
                f"augment_batch changed '{key}' with noise_std=0"
            )


# ---------------------------------------------------------------------------
# Tests 15-17: NeuroFusionCSVDataset
# ---------------------------------------------------------------------------


class TestNeuroFusionCSVDatasetADNI:
    """Tests for NeuroFusionCSVDataset in ADNI mode."""

    def test_adni_loads_correct_shapes(self, tmp_path: Path) -> None:
        """Test 15: ADNI dataset returns correct tensor shapes for all modalities."""
        from src.data.csv_dataset import NeuroFusionCSVDataset

        csv_path = str(tmp_path / "adni_train.csv")
        _write_adni_csv(csv_path, n_rows=20)

        ds = NeuroFusionCSVDataset(csv_path, mode="adni", fit_imputation=True)

        assert len(ds) == 20, f"Expected 20 samples, got {len(ds)}"

        item = ds[0]
        assert item["fluid"].shape == (6,), f"fluid shape: {item['fluid'].shape}"
        assert item["acoustic"].shape == (12,), f"acoustic shape: {item['acoustic'].shape}"
        assert item["motor"].shape == (8,), f"motor shape: {item['motor'].shape}"
        assert item["clinical"].shape == (10,), f"clinical shape: {item['clinical'].shape}"

    def test_adni_labels_have_correct_types(self, tmp_path: Path) -> None:
        """Test 16: ADNI labels are scalar float tensors."""
        from src.data.csv_dataset import NeuroFusionCSVDataset

        csv_path = str(tmp_path / "adni_train.csv")
        _write_adni_csv(csv_path, n_rows=10)

        ds = NeuroFusionCSVDataset(csv_path, mode="adni", fit_imputation=True)
        item = ds[0]

        for label_key in ("amyloid_label", "mmse_slope", "survival_time", "event_indicator"):
            assert label_key in item, f"Missing key: {label_key}"
            assert isinstance(item[label_key], torch.Tensor), f"'{label_key}' must be a tensor"
            assert item[label_key].dtype == torch.float32, f"'{label_key}' must be float32"

    def test_adni_patient_id_is_hashed(self, tmp_path: Path) -> None:
        """Test 17: Patient IDs are SHA-256 hashed (64 hex chars), not raw RID."""
        from src.data.csv_dataset import NeuroFusionCSVDataset

        csv_path = str(tmp_path / "adni_train.csv")
        _write_adni_csv(csv_path, n_rows=5)

        ds = NeuroFusionCSVDataset(csv_path, mode="adni", fit_imputation=True)
        item = ds[0]

        assert "patient_id" in item, "Missing 'patient_id' key"
        pid = item["patient_id"]
        assert isinstance(pid, str), "patient_id must be a string"
        assert len(pid) == 64, f"SHA-256 hash should be 64 chars, got {len(pid)}"
        # Must not be a simple numeric RID
        assert not pid.isdigit(), "patient_id should be hashed, not raw RID"


class TestNeuroFusionCSVDatasetBioHermes:
    """Tests for NeuroFusionCSVDataset in Bio-Hermes-001 mode."""

    def test_biohermes_loads_correct_shapes(self, tmp_path: Path) -> None:
        """Test 18: Bio-Hermes dataset returns correct tensor shapes."""
        from src.data.csv_dataset import NeuroFusionCSVDataset

        csv_path = str(tmp_path / "bh_train.csv")
        _write_biohermes_csv(csv_path, n_rows=15)

        ds = NeuroFusionCSVDataset(csv_path, mode="biohermes", fit_imputation=True)

        assert len(ds) == 15, f"Expected 15 samples, got {len(ds)}"
        item = ds[0]
        assert item["fluid"].shape == (6,), f"fluid shape: {item['fluid'].shape}"
        assert item["acoustic"].shape == (12,), f"acoustic shape (padded): {item['acoustic'].shape}"
        assert item["motor"].shape == (8,), f"motor shape: {item['motor'].shape}"
        assert item["clinical"].shape == (10,), f"clinical shape: {item['clinical'].shape}"

    def test_biohermes_acoustic_padded_to_12(self, tmp_path: Path) -> None:
        """Test 19: Bio-Hermes acoustic is padded from 10 to 12 with zeros."""
        from src.data.csv_dataset import NeuroFusionCSVDataset

        csv_path = str(tmp_path / "bh_train.csv")
        _write_biohermes_csv(csv_path, n_rows=10)

        ds = NeuroFusionCSVDataset(csv_path, mode="biohermes", fit_imputation=True)
        item = ds[0]

        # Positions 10 and 11 should be 0.0 (padding)
        assert item["acoustic"][10].item() == pytest.approx(0.0), (
            "acoustic pad position 10 should be 0.0"
        )
        assert item["acoustic"][11].item() == pytest.approx(0.0), (
            "acoustic pad position 11 should be 0.0"
        )

    def test_biohermes_survival_labels_all_nan(self, tmp_path: Path) -> None:
        """Test 20: Bio-Hermes survival labels are NaN (cross-sectional dataset)."""
        from src.data.csv_dataset import NeuroFusionCSVDataset

        csv_path = str(tmp_path / "bh_train.csv")
        _write_biohermes_csv(csv_path, n_rows=10)

        ds = NeuroFusionCSVDataset(csv_path, mode="biohermes", fit_imputation=True)

        for i in range(len(ds)):
            item = ds[i]
            assert math.isnan(item["mmse_slope"].item()), (
                f"BH mmse_slope at idx {i} should be NaN"
            )
            assert math.isnan(item["survival_time"].item()), (
                f"BH survival_time at idx {i} should be NaN"
            )
            assert math.isnan(item["event_indicator"].item()), (
                f"BH event_indicator at idx {i} should be NaN"
            )


class TestNeuroFusionCSVDatasetImputation:
    """Tests for NeuroFusionCSVDataset NaN imputation."""

    def test_adni_no_nan_in_feature_tensors_after_imputation(self, tmp_path: Path) -> None:
        """Test 21: After imputation, no NaN values remain in feature tensors."""
        from src.data.csv_dataset import NeuroFusionCSVDataset

        csv_path = str(tmp_path / "adni_train.csv")
        _write_adni_csv(csv_path, n_rows=30, include_nans=True)

        ds = NeuroFusionCSVDataset(csv_path, mode="adni", fit_imputation=True)

        for i in range(len(ds)):
            item = ds[i]
            for key in ("fluid", "acoustic", "motor", "clinical"):
                assert not torch.isnan(item[key]).any(), (
                    f"NaN found in '{key}' tensor at idx {i} after imputation"
                )

    def test_val_uses_train_imputation_stats(self, tmp_path: Path) -> None:
        """Test 22: Val split applies train-fit imputation stats (not its own)."""
        from src.data.csv_dataset import NeuroFusionCSVDataset

        train_path = str(tmp_path / "adni_train.csv")
        val_path = str(tmp_path / "adni_val.csv")
        _write_adni_csv(train_path, n_rows=20, include_nans=True)
        _write_adni_csv(val_path, n_rows=5, include_nans=True)

        train_ds = NeuroFusionCSVDataset(train_path, mode="adni", fit_imputation=True)
        val_ds = NeuroFusionCSVDataset(
            val_path,
            mode="adni",
            fit_imputation=False,
            imputation_stats=train_ds.imputation_stats,
        )

        # Val dataset must exist and have no NaN in features
        assert len(val_ds) > 0
        for i in range(len(val_ds)):
            item = val_ds[i]
            for key in ("fluid", "acoustic", "motor", "clinical"):
                assert not torch.isnan(item[key]).any(), (
                    f"NaN in val '{key}' at idx {i}"
                )


# ---------------------------------------------------------------------------
# Tests 23-24: NeuroFusionTrainer
# ---------------------------------------------------------------------------


class TestNeuroFusionTrainer:
    """Tests for NeuroFusionTrainer.train_epoch and early stopping."""

    def _build_dummy_loader(self, batch_size: int = 4, n_batches: int = 3) -> "DataLoader":
        """Build a minimal DataLoader from synthetic tensors.

        Args:
            batch_size: Samples per batch.
            n_batches: Total number of batches.

        Returns:
            DataLoader yielding dict batches.
        """
        from torch.utils.data import DataLoader, Dataset

        class _SyntheticDataset(Dataset):
            def __init__(self, n: int):
                self.n = n

            def __len__(self) -> int:
                return self.n

            def __getitem__(self, idx: int) -> dict:
                return {
                    "fluid": torch.randn(6),
                    "acoustic": torch.randn(12),
                    "motor": torch.randn(8),
                    "clinical": torch.randn(10),
                    "amyloid_label": torch.tensor(float(idx % 2)),
                    "mmse_slope": torch.tensor(-0.5),
                    "survival_time": torch.tensor(12.0),
                    "event_indicator": torch.tensor(float(idx % 2)),
                    "patient_id": "aaaa" * 16,  # dummy 64-char hash
                }

        ds = _SyntheticDataset(batch_size * n_batches)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    def test_train_epoch_runs_without_error_on_cpu(self) -> None:
        """Test 23: NeuroFusionTrainer.train_epoch completes on CPU without error."""
        from src.training.trainer import NeuroFusionTrainer

        model = _make_minimal_model()
        config = {
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "gradient_accumulation_steps": 1,
            "n_epochs": 2,
            "batch_size": 4,
            "early_stopping_patience": 5,
            "onecycle_pct_start": 0.3,
            "augmentation_noise_std": 0.01,
            "wandb_enabled": False,
        }
        trainer = NeuroFusionTrainer(model, config, device="cpu")
        loader = self._build_dummy_loader(batch_size=4, n_batches=3)
        loss_fn = MultiTaskLoss({"cls": 1.0, "reg": 0.5, "surv": 0.5})

        result = trainer.train_epoch(loader, loss_fn)

        assert "loss" in result, "train_epoch result must have 'loss' key"
        assert math.isfinite(result["loss"]), f"train_epoch loss is not finite: {result['loss']}"
        assert result["loss"] > 0.0, "train_epoch loss should be positive"

    def test_early_stopping_triggers(self) -> None:
        """Test 24: NeuroFusionTrainer.fit early stops when val_auc does not improve."""
        from src.training.trainer import NeuroFusionTrainer

        model = _make_minimal_model()
        config = {
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "gradient_accumulation_steps": 1,
            "n_epochs": 100,  # Would run 100 epochs without early stopping
            "batch_size": 4,
            "early_stopping_patience": 3,  # Stop after 3 epochs of no improvement
            "onecycle_pct_start": 0.3,
            "augmentation_noise_std": 0.0,
            "wandb_enabled": False,
        }

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = NeuroFusionTrainer(model, config, device="cpu")
            loader = self._build_dummy_loader(batch_size=8, n_batches=2)
            loss_fn = MultiTaskLoss({"cls": 1.0, "reg": 0.0, "surv": 0.0})

            results = trainer.fit(
                train_loader=loader,
                val_loader=loader,
                loss_fn=loss_fn,
                n_epochs=100,
                checkpoint_dir=tmpdir,
                save_every_n_epochs=50,
            )

            # Must have early-stopped before 100 epochs
            total_epochs = len(results["history"])
            assert total_epochs < 100, (
                f"Expected early stopping before epoch 100, but ran {total_epochs} epochs"
            )

    def test_trainer_evaluate_returns_metrics(self) -> None:
        """Test 25: NeuroFusionTrainer.evaluate returns dict with expected keys."""
        from src.training.trainer import NeuroFusionTrainer

        model = _make_minimal_model()
        config = {
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "gradient_accumulation_steps": 1,
            "n_epochs": 1,
            "batch_size": 4,
            "early_stopping_patience": 5,
            "onecycle_pct_start": 0.3,
            "augmentation_noise_std": 0.0,
            "wandb_enabled": False,
        }
        trainer = NeuroFusionTrainer(model, config, device="cpu")
        loader = self._build_dummy_loader(batch_size=4, n_batches=3)
        loss_fn = MultiTaskLoss({"cls": 1.0, "reg": 0.0, "surv": 0.0})

        result = trainer.evaluate(loader, loss_fn)

        assert "loss" in result, "evaluate result must have 'loss' key"
        assert "auc" in result, "evaluate result must have 'auc' key"
        assert "rmse" in result, "evaluate result must have 'rmse' key"
        assert "c_index" in result, "evaluate result must have 'c_index' key"
        assert math.isfinite(result["loss"]), f"evaluate loss not finite: {result['loss']}"
