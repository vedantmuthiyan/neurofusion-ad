"""Cross-modal attention fusion for NeuroFusion-AD.

Fuses four modality embeddings (fluid, acoustic, motor, clinical) using
multi-head self-attention over the modality sequence dimension.
Fluid biomarker embedding acts as the query anchor.

Architecture:
    - Input: 4 embeddings, each [batch, 768] -> stack to [batch, 4, 768]
    - Multi-head self-attention: nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)
    - Residual connection + LayerNorm
    - Feed-forward: Linear(768, 3072) -> GELU -> Dropout(0.1) -> Linear(3072, 768)
    - Residual connection + LayerNorm
    - Output: [batch, 768] via mean-pooling over the 4 modality tokens

Document traceability:
    SAD-001 § 5.2 — Cross-Modal Attention Fusion
    SRS-001 § 4.3 — Multimodal Integration Requirements
"""

from __future__ import annotations

import torch
import torch.nn as nn
import structlog

log = structlog.get_logger(__name__)


class CrossModalAttention(nn.Module):
    """Cross-modal attention fusion module.

    Stacks four 768-dim modality embeddings into a sequence of 4 tokens
    and applies transformer-style multi-head attention to learn cross-modal
    dependencies. The fused representation is the mean over all 4 tokens.

    The fluid biomarker embedding is placed at position 0 (index 0) of the
    sequence, acting as the query anchor per SAD-001 § 5.2.1.

    Attributes:
        embed_dim: Embedding dimension (768).
        num_heads: Number of attention heads (8).
        attention: nn.MultiheadAttention module.
        attn_norm: LayerNorm applied after attention residual.
        ff_norm: LayerNorm applied after feed-forward residual.
        ff: Feed-forward network (768 -> 3072 -> 768).
        dropout: Dropout(0.1) used in feed-forward.

    Args:
        embed_dim: Embedding dimension. Must be divisible by num_heads.
        num_heads: Number of attention heads.
        ff_dim: Feed-forward hidden dimension.
        dropout: Dropout probability.

    Example:
        >>> fusion = CrossModalAttention()
        >>> fluid = torch.randn(4, 768)
        >>> acoustic = torch.randn(4, 768)
        >>> motor = torch.randn(4, 768)
        >>> clinical = torch.randn(4, 768)
        >>> out = fusion(fluid, acoustic, motor, clinical)
        >>> out.shape
        torch.Size([4, 768])
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        ff_dim: int = 3072,
        dropout: float = 0.1,
    ) -> None:
        """Initialise CrossModalAttention.

        Args:
            embed_dim: Embedding dimension (default 768).
            num_heads: Number of attention heads (default 8). Must divide embed_dim.
            ff_dim: Feed-forward hidden dimension (default 3072 = 4 * 768).
            dropout: Dropout probability (default 0.1).

        Raises:
            ValueError: If embed_dim is not divisible by num_heads.
        """
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}). "
                f"Remainder: {embed_dim % num_heads}."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Multi-head self-attention with batch_first=True so input shape is
        # [batch, seq_len, embed_dim] (SAD-001 § 5.2.1)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(embed_dim)

        # Feed-forward network: Linear(768 -> 3072) -> GELU -> Dropout -> Linear(3072 -> 768)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.ff_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout)

        log.info(
            "CrossModalAttention initialised",
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )

    def forward(
        self,
        fluid: torch.Tensor,
        acoustic: torch.Tensor,
        motor: torch.Tensor,
        clinical: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse four modality embeddings via multi-head attention.

        Fluid biomarker embedding is placed at position 0 of the token sequence
        as the query anchor (SAD-001 § 5.2.1). All four modalities participate
        symmetrically as keys and values; the query anchor position ensures the
        fused representation is grounded in amyloid/tau signal.

        Args:
            fluid: Fluid biomarker embedding [batch, 768]. Query anchor.
            acoustic: Acoustic embedding [batch, 768].
            motor: Motor embedding [batch, 768].
            clinical: Clinical/demographic embedding [batch, 768].

        Returns:
            Fused representation [batch, 768] — mean-pooled over 4 modality tokens
            after two transformer sub-layers (attention + feed-forward).

        Raises:
            ValueError: If any input tensor has wrong shape.
        """
        # Validate input shapes
        for name, tensor in (
            ("fluid", fluid),
            ("acoustic", acoustic),
            ("motor", motor),
            ("clinical", clinical),
        ):
            if tensor.dim() != 2 or tensor.shape[1] != self.embed_dim:
                raise ValueError(
                    f"Input '{name}' must have shape [batch, {self.embed_dim}], "
                    f"got {list(tensor.shape)}."
                )

        # Stack to [batch, 4, embed_dim]; fluid is placed at index 0 as query anchor
        x = torch.stack([fluid, acoustic, motor, clinical], dim=1)

        # Sub-layer 1: Multi-head self-attention + residual + LayerNorm
        attn_output, _ = self.attention(x, x, x)
        x = self.attn_norm(x + attn_output)

        # Sub-layer 2: Feed-forward network + residual + LayerNorm
        ff_output = self.ff(x)
        x = self.ff_norm(x + ff_output)

        # Mean-pool over the 4 modality tokens -> [batch, 768]
        fused: torch.Tensor = x.mean(dim=1)

        log.debug(
            "CrossModalAttention forward complete",
            batch_size=fluid.shape[0],
            output_shape=list(fused.shape),
        )
        return fused
