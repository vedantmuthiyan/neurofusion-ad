"""Unit tests for the CrossModalAttention fusion module.

Tests are self-contained — only torch is used for synthetic tensor generation.
No real patient data (PHI) is used in any test.

Test coverage:
    test_output_shape             : output tensor has shape [batch, 768]
    test_output_type              : output dtype is torch.float32
    test_no_nan_output            : output contains no NaN values
    test_fluid_is_anchor          : fluid at position 0; other modalities zeroed
    test_invalid_embed_dim_raises : embed_dim not divisible by num_heads raises ValueError
    test_different_batch_sizes    : batch_size 1, 8, 16 all produce correct shapes
    test_training_vs_eval_mode    : dropout is active in train, inactive in eval

Document traceability:
    SAD-001 § 5.2 — Cross-Modal Attention Fusion
    SRS-001 § 4.3 — Multimodal Integration Requirements
    IEC 62304 § 5.5.5 — Software unit verification
"""

from __future__ import annotations

import pytest
import torch

from src.models.cross_modal_attention import CrossModalAttention

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBED_DIM = 768
DEFAULT_BATCH_SIZE = 4


# ---------------------------------------------------------------------------
# Shared helpers — pure torch, no numpy (avoids NumPy 2.x / torch ABI issues)
# ---------------------------------------------------------------------------

def _make_embeddings(batch_size: int = DEFAULT_BATCH_SIZE) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Return four random float32 embeddings of shape [batch_size, 768].

    Uses a fixed seed for reproducibility within each test invocation.

    Args:
        batch_size: Number of samples in the batch.

    Returns:
        Tuple of (fluid, acoustic, motor, clinical) tensors, each [batch_size, 768].
    """
    torch.manual_seed(0)
    fluid = torch.randn(batch_size, EMBED_DIM, dtype=torch.float32)
    acoustic = torch.randn(batch_size, EMBED_DIM, dtype=torch.float32)
    motor = torch.randn(batch_size, EMBED_DIM, dtype=torch.float32)
    clinical = torch.randn(batch_size, EMBED_DIM, dtype=torch.float32)
    return fluid, acoustic, motor, clinical


# ===========================================================================
# CrossModalAttention unit tests
# ===========================================================================

class TestCrossModalAttention:
    """Unit tests for CrossModalAttention (SAD-001 § 5.2)."""

    @pytest.fixture
    def fusion(self) -> CrossModalAttention:
        """Provide a freshly initialised CrossModalAttention module in eval mode.

        Returns:
            CrossModalAttention instance set to eval mode.
        """
        model = CrossModalAttention()
        model.eval()
        return model

    def test_output_shape(self, fusion: CrossModalAttention) -> None:
        """Output tensor must have shape [batch_size, 768].

        Verifies that mean-pooling over the 4 modality tokens collapses the
        sequence dimension and produces a single fused vector per sample.

        Args:
            fusion: CrossModalAttention fixture.
        """
        fluid, acoustic, motor, clinical = _make_embeddings(DEFAULT_BATCH_SIZE)
        with torch.no_grad():
            out = fusion(fluid, acoustic, motor, clinical)
        assert out.shape == (DEFAULT_BATCH_SIZE, EMBED_DIM), (
            f"Expected shape ({DEFAULT_BATCH_SIZE}, {EMBED_DIM}), got {out.shape}"
        )

    def test_output_type(self, fusion: CrossModalAttention) -> None:
        """Output tensor must be of dtype torch.float32.

        Args:
            fusion: CrossModalAttention fixture.
        """
        fluid, acoustic, motor, clinical = _make_embeddings(DEFAULT_BATCH_SIZE)
        with torch.no_grad():
            out = fusion(fluid, acoustic, motor, clinical)
        assert out.dtype == torch.float32, (
            f"Expected torch.float32, got {out.dtype}"
        )

    def test_no_nan_output(self, fusion: CrossModalAttention) -> None:
        """Output must contain no NaN values for random finite inputs.

        Args:
            fusion: CrossModalAttention fixture.
        """
        fluid, acoustic, motor, clinical = _make_embeddings(DEFAULT_BATCH_SIZE)
        with torch.no_grad():
            out = fusion(fluid, acoustic, motor, clinical)
        assert not torch.isnan(out).any(), (
            "CrossModalAttention produced NaN values for valid random inputs."
        )

    def test_fluid_is_anchor(self, fusion: CrossModalAttention) -> None:
        """Fluid embedding at position 0 with all other modalities zeroed must
        produce a valid (non-NaN, correct-shape) output.

        This exercises the query-anchor property: even when acoustic, motor,
        and clinical embeddings carry no signal, the fluid embedding at index 0
        provides a well-defined query for the attention mechanism.

        Args:
            fusion: CrossModalAttention fixture.
        """
        torch.manual_seed(1)
        fluid = torch.randn(DEFAULT_BATCH_SIZE, EMBED_DIM, dtype=torch.float32)
        zeros = torch.zeros(DEFAULT_BATCH_SIZE, EMBED_DIM, dtype=torch.float32)

        with torch.no_grad():
            out = fusion(fluid, zeros, zeros, zeros)

        assert out.shape == (DEFAULT_BATCH_SIZE, EMBED_DIM), (
            f"Expected shape ({DEFAULT_BATCH_SIZE}, {EMBED_DIM}), got {out.shape}"
        )
        assert not torch.isnan(out).any(), (
            "Output contains NaN when acoustic, motor, clinical are zeroed."
        )

    def test_invalid_embed_dim_raises(self) -> None:
        """Constructing CrossModalAttention with embed_dim not divisible by
        num_heads must raise ValueError.

        embed_dim=100, num_heads=8 → 100 % 8 = 4 ≠ 0.

        Raises:
            ValueError: Expected from CrossModalAttention.__init__.
        """
        with pytest.raises(ValueError, match="divisible"):
            CrossModalAttention(embed_dim=100, num_heads=8)

    @pytest.mark.parametrize("batch_size", [1, 8, 16])
    def test_different_batch_sizes(self, batch_size: int) -> None:
        """CrossModalAttention must produce shape [batch_size, 768] for
        batch sizes 1, 8, and 16.

        Args:
            batch_size: Number of samples in the batch (parameterised).
        """
        model = CrossModalAttention()
        model.eval()
        fluid, acoustic, motor, clinical = _make_embeddings(batch_size)
        with torch.no_grad():
            out = model(fluid, acoustic, motor, clinical)
        assert out.shape == (batch_size, EMBED_DIM), (
            f"Expected shape ({batch_size}, {EMBED_DIM}), got {out.shape}"
        )

    def test_training_vs_eval_mode(self) -> None:
        """Dropout must be active in train mode and inactive in eval mode.

        In train mode two successive forward passes through the same model
        with the same inputs should (with high probability) produce different
        outputs due to stochastic dropout. In eval mode the outputs must be
        identical (deterministic).

        Note:
            This test uses a batch_size of 16 and a high-variance input to
            maximise the probability that dropout produces a measurable
            difference.  It relies on the default dropout=0.1 applied inside
            the feed-forward sub-layer of CrossModalAttention.
        """
        torch.manual_seed(42)
        model = CrossModalAttention()

        fluid = torch.randn(16, EMBED_DIM, dtype=torch.float32)
        acoustic = torch.randn(16, EMBED_DIM, dtype=torch.float32)
        motor = torch.randn(16, EMBED_DIM, dtype=torch.float32)
        clinical = torch.randn(16, EMBED_DIM, dtype=torch.float32)

        # Eval mode: two passes must be identical (dropout disabled)
        model.eval()
        with torch.no_grad():
            out_eval_1 = model(fluid, acoustic, motor, clinical)
            out_eval_2 = model(fluid, acoustic, motor, clinical)
        assert torch.allclose(out_eval_1, out_eval_2), (
            "CrossModalAttention is non-deterministic in eval mode "
            "(dropout should be disabled)."
        )

        # Train mode: two passes should differ due to stochastic dropout
        model.train()
        # Do not use torch.no_grad() so that dropout is active
        out_train_1 = model(fluid, acoustic, motor, clinical)
        out_train_2 = model(fluid, acoustic, motor, clinical)
        assert not torch.allclose(out_train_1, out_train_2), (
            "CrossModalAttention produced identical outputs in train mode. "
            "Dropout may not be active."
        )
