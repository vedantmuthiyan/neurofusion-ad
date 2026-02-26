"""Unit tests for NeuroFusionAD integrated model.

Tests cover model instantiation, forward pass output shapes, NaN checks,
disclaimer presence, parameter count, eval mode, error handling for missing
keys, and multiple batch sizes.

IEC 62304 Traceability:
    SAD-001 § 5.4 — Full Model Architecture
    SRS-001 § 3.1 — System Overview
"""

import pytest
import torch

from src.models.neurofusion_model import NeuroFusionAD, CLINICAL_DISCLAIMER


# ---------------------------------------------------------------------------
# Test fixture helpers — never use torch.from_numpy() or numpy arrays
# ---------------------------------------------------------------------------


def make_valid_batch(batch_size: int) -> dict[str, torch.Tensor]:
    """Create a valid synthetic batch with all range constraints satisfied.

    Constructs tensors using torch.zeros/torch.randn (never numpy).
    All validated biomarker features are set to values within their
    physiological ranges per SRS-001 § 4.2.

    Args:
        batch_size: Number of patient samples in the batch.

    Returns:
        Dict with keys 'fluid', 'acoustic', 'motor', 'clinical' as float tensors.
    """
    fluid = torch.zeros(batch_size, 6)
    fluid[:, 0] = 5.0     # pTau-217 in [0.1, 100]
    fluid[:, 1] = 0.1     # Abeta42/40 ratio in [0.01, 0.30]
    fluid[:, 2] = 50.0    # NfL in [5, 200]

    acoustic = torch.zeros(batch_size, 12)
    acoustic[:, 0] = 0.005   # jitter in [0.0001, 0.05]
    acoustic[:, 1] = 0.05    # shimmer in [0.001, 0.30]

    motor = torch.randn(batch_size, 8)

    clinical = torch.zeros(batch_size, 10)
    clinical[:, 3] = 25.0    # MMSE in [0, 30]

    return {
        "fluid": fluid,
        "acoustic": acoustic,
        "motor": motor,
        "clinical": clinical,
    }


# ---------------------------------------------------------------------------
# Test 1 — Model instantiation
# ---------------------------------------------------------------------------


def test_model_instantiation() -> None:
    """NeuroFusionAD can be instantiated with default arguments without error."""
    model = NeuroFusionAD()
    assert model is not None
    assert isinstance(model, torch.nn.Module)


# ---------------------------------------------------------------------------
# Test 2 — Forward output keys
# ---------------------------------------------------------------------------


def test_forward_output_keys() -> None:
    """Forward pass returns a dict with all 5 required output keys."""
    model = NeuroFusionAD()
    model.eval()
    batch = make_valid_batch(batch_size=4)
    with torch.no_grad():
        outputs = model(batch)
    expected_keys = {
        "amyloid_logit",
        "mmse_slope",
        "cox_log_hazard",
        "fused_embedding",
        "disclaimer",
    }
    assert set(outputs.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Test 3 — amyloid_logit shape
# ---------------------------------------------------------------------------


def test_forward_amyloid_shape() -> None:
    """amyloid_logit output has shape [batch_size, 1]."""
    model = NeuroFusionAD()
    model.eval()
    batch = make_valid_batch(batch_size=8)
    with torch.no_grad():
        outputs = model(batch)
    assert outputs["amyloid_logit"].shape == (8, 1), (
        f"Expected (8, 1), got {outputs['amyloid_logit'].shape}"
    )


# ---------------------------------------------------------------------------
# Test 4 — mmse_slope shape
# ---------------------------------------------------------------------------


def test_forward_mmse_slope_shape() -> None:
    """mmse_slope output has shape [batch_size, 1]."""
    model = NeuroFusionAD()
    model.eval()
    batch = make_valid_batch(batch_size=8)
    with torch.no_grad():
        outputs = model(batch)
    assert outputs["mmse_slope"].shape == (8, 1), (
        f"Expected (8, 1), got {outputs['mmse_slope'].shape}"
    )


# ---------------------------------------------------------------------------
# Test 5 — cox_log_hazard shape
# ---------------------------------------------------------------------------


def test_forward_cox_hazard_shape() -> None:
    """cox_log_hazard output has shape [batch_size, 1]."""
    model = NeuroFusionAD()
    model.eval()
    batch = make_valid_batch(batch_size=8)
    with torch.no_grad():
        outputs = model(batch)
    assert outputs["cox_log_hazard"].shape == (8, 1), (
        f"Expected (8, 1), got {outputs['cox_log_hazard'].shape}"
    )


# ---------------------------------------------------------------------------
# Test 6 — fused_embedding shape
# ---------------------------------------------------------------------------


def test_forward_embedding_shape() -> None:
    """fused_embedding output has shape [batch_size, 768]."""
    model = NeuroFusionAD()
    model.eval()
    batch = make_valid_batch(batch_size=8)
    with torch.no_grad():
        outputs = model(batch)
    assert outputs["fused_embedding"].shape == (8, 768), (
        f"Expected (8, 768), got {outputs['fused_embedding'].shape}"
    )


# ---------------------------------------------------------------------------
# Test 7 — No NaN in any output tensor
# ---------------------------------------------------------------------------


def test_forward_no_nan() -> None:
    """No output tensor contains NaN values after a forward pass."""
    model = NeuroFusionAD()
    model.eval()
    batch = make_valid_batch(batch_size=16)
    with torch.no_grad():
        outputs = model(batch)
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            assert not torch.isnan(val).any(), (
                f"NaN detected in output '{key}' after forward pass."
            )


# ---------------------------------------------------------------------------
# Test 8 — Disclaimer present and exact
# ---------------------------------------------------------------------------


def test_disclaimer_present() -> None:
    """Disclaimer string in outputs matches the mandatory regulatory text exactly."""
    model = NeuroFusionAD()
    model.eval()
    batch = make_valid_batch(batch_size=4)
    with torch.no_grad():
        outputs = model(batch)
    assert "disclaimer" in outputs
    assert outputs["disclaimer"] == CLINICAL_DISCLAIMER
    assert outputs["disclaimer"] == (
        "This tool is intended to support, not replace, clinical judgment."
    )


# ---------------------------------------------------------------------------
# Test 9 — Parameter count > 10M
# ---------------------------------------------------------------------------


def test_count_parameters() -> None:
    """Model has more than 10M trainable parameters (target ~60M)."""
    model = NeuroFusionAD()
    n_params = model.count_parameters()
    assert n_params > 10_000_000, (
        f"Expected > 10M parameters, got {n_params:,}. "
        "Check that all sub-modules are registered correctly."
    )


# ---------------------------------------------------------------------------
# Test 10 — Eval mode does not raise
# ---------------------------------------------------------------------------


def test_model_eval_mode() -> None:
    """Model runs forward pass without error in eval mode."""
    model = NeuroFusionAD()
    model.eval()
    batch = make_valid_batch(batch_size=4)
    with torch.no_grad():
        outputs = model(batch)
    # All output tensors should be finite
    for key, val in outputs.items():
        if isinstance(val, torch.Tensor):
            assert torch.isfinite(val).all(), (
                f"Non-finite values in '{key}' during eval mode."
            )


# ---------------------------------------------------------------------------
# Test 11 — Missing 'fluid' key raises KeyError
# ---------------------------------------------------------------------------


def test_missing_key_raises() -> None:
    """KeyError is raised when 'fluid' key is missing from the input batch."""
    model = NeuroFusionAD()
    model.eval()
    batch = make_valid_batch(batch_size=4)
    del batch["fluid"]  # Remove required key
    with pytest.raises(KeyError, match="fluid"):
        model(batch)


# ---------------------------------------------------------------------------
# Test 12 — Different batch sizes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_different_batch_sizes(batch_size: int) -> None:
    """Model runs correctly for batch_size=1, 4, and 16."""
    model = NeuroFusionAD()
    model.eval()
    batch = make_valid_batch(batch_size=batch_size)
    with torch.no_grad():
        outputs = model(batch)
    assert outputs["amyloid_logit"].shape == (batch_size, 1)
    assert outputs["mmse_slope"].shape == (batch_size, 1)
    assert outputs["cox_log_hazard"].shape == (batch_size, 1)
    assert outputs["fused_embedding"].shape == (batch_size, 768)
    assert outputs["disclaimer"] == CLINICAL_DISCLAIMER
