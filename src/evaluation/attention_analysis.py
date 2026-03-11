"""Cross-modal attention weight extraction and visualization for NeuroFusion-AD.

Registers forward hooks on CrossModalAttention to capture multi-head
attention weights during inference, then aggregates and visualizes them
to show which modalities the model attends to most.

This analysis provides regulatory-required model interpretability per
SRS-001 § 6.4 (Explainability Requirements).

IEC 62304 compliance:
    - Hooks are always removed after extraction to prevent memory leaks.
    - No PHI is captured or logged.

Document traceability:
    SRS-001 § 6.4 — Explainability and Interpretability Requirements
    SAD-001 § 5.2 — Cross-Modal Attention Fusion
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import structlog

log = structlog.get_logger(__name__)

MODALITY_NAMES = ["fluid", "acoustic", "motor", "clinical"]


class AttentionAnalyzer:
    """Extract and visualize cross-modal attention weights.

    CrossModalAttention stacks modalities as [fluid, acoustic, motor, clinical]
    at indices [0, 1, 2, 3]. Attention weights show how each modality attends
    to every other modality.

    Since fluid is the query anchor (index 0), we focus on fluid's attention
    distribution across all 4 modalities.

    The analyzer uses PyTorch forward hooks to capture attention weights
    from nn.MultiheadAttention without modifying model code.

    Attributes:
        modality_names: Fixed list ['fluid', 'acoustic', 'motor', 'clinical'].

    Example:
        >>> analyzer = AttentionAnalyzer()
        >>> results = analyzer.extract_attention_weights(model, dataloader)
        >>> importance = analyzer.get_modality_importance_scores(results)
    """

    def __init__(self) -> None:
        """Initialise AttentionAnalyzer."""
        self.modality_names = MODALITY_NAMES

    def extract_attention_weights(
        self,
        model: nn.Module,
        dataloader,
    ) -> dict:
        """Register forward hooks on CrossModalAttention to capture attention weights.

        Hooks are registered on the model's CrossModalAttention.attention
        (nn.MultiheadAttention) module. The attention_weights returned by
        the module's forward() are captured for each batch.

        Args:
            model: NeuroFusionAD instance. Must have a .fusion attribute
                which is a CrossModalAttention with an .attention attribute.
            dataloader: PyTorch DataLoader yielding batches with keys
                'fluid', 'acoustic', 'motor', 'clinical'.

        Returns:
            Dict with:
                {
                    'attention_weights': ndarray[n_samples, n_heads, 4, 4],
                    'modality_importance': ndarray[4],
                    'modality_names': ['fluid', 'acoustic', 'motor', 'clinical'],
                }
        """
        captured_weights: list[np.ndarray] = []

        def _hook_fn(
            module: nn.Module,
            input: tuple,
            output: tuple,
        ) -> None:
            """Hook function that captures attention weights.

            nn.MultiheadAttention returns (attn_output, attn_weights).
            attn_weights has shape [batch, n_heads, seq, seq] when
            need_weights=True and average_attn_weights=False, or
            [batch, seq, seq] when average_attn_weights=True (default).

            We force need_weights=True + average_attn_weights=False to
            get per-head weights by monkey-patching the forward call below.
            """
            # output is (attn_output, attn_weights) or just attn_output
            if isinstance(output, tuple) and len(output) == 2:
                _, weights = output
                if weights is not None:
                    captured_weights.append(
                        weights.detach().cpu().numpy()
                    )

        # Register hook on CrossModalAttention.attention
        if not hasattr(model, "fusion") or not hasattr(model.fusion, "attention"):
            raise AttributeError(
                "model.fusion.attention not found. "
                "Expected NeuroFusionAD with a CrossModalAttention fusion module."
            )

        # We need to temporarily patch the CrossModalAttention.forward to
        # request attention weights, since the default forward() discards them.
        original_forward = model.fusion.forward

        def _patched_fusion_forward(fluid, acoustic, motor, clinical):
            """Patched forward that captures attention weights."""
            x = torch.stack([fluid, acoustic, motor, clinical], dim=1)
            attn_output, attn_weights = model.fusion.attention(
                x, x, x,
                need_weights=True,
                average_attn_weights=False,
            )
            # attn_weights: [batch, n_heads, seq_len, seq_len]
            if attn_weights is not None:
                captured_weights.append(attn_weights.detach().cpu().numpy())
            # Continue with normal forward
            x_res = model.fusion.attn_norm(x + attn_output)
            ff_output = model.fusion.ff(x_res)
            x_res = model.fusion.ff_norm(x_res + ff_output)
            return x_res.mean(dim=1)

        model.fusion.forward = _patched_fusion_forward  # type: ignore[method-assign]

        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                try:
                    _ = model(batch)
                except Exception as exc:
                    log.warning(
                        "AttentionAnalyzer: batch failed",
                        error=str(exc),
                    )
                    continue

        # Restore original forward
        model.fusion.forward = original_forward  # type: ignore[method-assign]

        if not captured_weights:
            log.warning("AttentionAnalyzer: no attention weights captured")
            return {
                "attention_weights": np.zeros((0, 8, 4, 4)),
                "modality_importance": np.full(4, float("nan")),
                "modality_names": self.modality_names,
            }

        # Stack all captured weights: [total_samples, n_heads, 4, 4]
        all_weights = np.concatenate(captured_weights, axis=0)

        # Compute per-modality importance as mean attention received (column mean)
        # Mean over samples, heads, and query positions -> [4] modality importance
        modality_importance = all_weights.mean(axis=(0, 1, 2))  # [4]

        # Normalize to sum to 1
        total = modality_importance.sum()
        if total > 0:
            modality_importance = modality_importance / total

        log.info(
            "AttentionAnalyzer: weights extracted",
            n_samples=all_weights.shape[0],
            n_heads=all_weights.shape[1],
            modality_importance=modality_importance.tolist(),
        )

        return {
            "attention_weights": all_weights,
            "modality_importance": modality_importance,
            "modality_names": self.modality_names,
        }

    def plot_attention_heatmap(
        self,
        attention_results: dict,
        save_path: str,
    ) -> None:
        """Save attention heatmap (4x4 average across samples and heads).

        Rows = query modality, Columns = key modality.
        Cell (i, j) = mean attention from modality i to modality j.

        Args:
            attention_results: Dict returned by extract_attention_weights().
            save_path: Absolute path to save the PNG figure.
        """
        weights = attention_results["attention_weights"]  # [N, n_heads, 4, 4]
        modality_names = attention_results.get("modality_names", self.modality_names)

        if weights.shape[0] == 0:
            log.warning("AttentionAnalyzer: empty weights, skipping heatmap")
            return

        # Average over samples and heads: [4, 4]
        mean_attn = weights.mean(axis=(0, 1))

        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(mean_attn, cmap="Blues", vmin=0, vmax=mean_attn.max())

        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(
            [f"Key: {m}" for m in modality_names], rotation=30, ha="right", fontsize=10
        )
        ax.set_yticklabels(
            [f"Query: {m}" for m in modality_names], fontsize=10
        )

        # Annotate cells
        for i in range(4):
            for j in range(4):
                ax.text(
                    j, i,
                    f"{mean_attn[i, j]:.3f}",
                    ha="center", va="center",
                    fontsize=9,
                    color="black" if mean_attn[i, j] < 0.6 * mean_attn.max() else "white",
                )

        plt.colorbar(im, ax=ax, label="Mean Attention Weight")
        ax.set_title(
            "Cross-Modal Attention Heatmap\n(Mean over samples and heads)",
            fontsize=13,
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        log.info("Attention heatmap saved", path=save_path)

    def get_modality_importance_scores(
        self,
        attention_results: dict,
    ) -> dict:
        """Return modality importance as normalized attention weights.

        Importance is computed as the mean attention weight received by each
        modality (averaged over queries, heads, and samples).

        Args:
            attention_results: Dict returned by extract_attention_weights().

        Returns:
            Dict mapping modality name to importance score. Scores sum to ~1.0.
            Example: {'fluid': 0.31, 'acoustic': 0.24, 'motor': 0.22, 'clinical': 0.23}
        """
        modality_importance = attention_results["modality_importance"]  # [4]
        modality_names = attention_results.get("modality_names", self.modality_names)

        scores = {
            name: float(imp)
            for name, imp in zip(modality_names, modality_importance)
        }

        log.info("Modality importance scores", scores=scores)
        return scores
