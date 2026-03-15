#!/usr/bin/env python3
"""Run SHAP analysis on ADNI test set.

Loads best ADNI model from models/final/best_model.pth.
Uses KernelExplainer (model-agnostic, regulatory-compliant).

Background dataset: 50 samples from ADNI train (or synthetic data in dry run).
Test set: first 20 samples from ADNI test (or synthetic data in dry run).

Produces:
    docs/figures/shap_summary.png
    docs/figures/shap_waterfall_0.png  (sample with highest amyloid probability)
    docs/figures/shap_waterfall_1.png  (sample with lowest amyloid probability)
    docs/figures/shap_waterfall_2.png  (sample closest to decision boundary 0.5)

Usage:
    python scripts/run_shap.py [--checkpoint PATH] [--dry-run]

Note: KernelExplainer is slow (O(n_features * nsamples) model calls).
For 36 features and nsamples=200, expect ~7200 forward passes.
On CPU with batch_size=1 this takes ~5 minutes for 20 samples.

IEC 62304 requirement traceability:
    SRS-001 § 6.4 — Explainability Requirements
    SAD-001 § 5.5 — SHAP Explainability Layer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import structlog

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.neurofusion_model import NeuroFusionAD
from src.data.dataset import generate_synthetic_adni
from src.data.csv_dataset import NeuroFusionCSVDataset
from src.evaluation.shap_explainability import NeuralFusionSHAPExplainer

log = structlog.get_logger(__name__)

FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"


def _find_interesting_samples(predictions: np.ndarray) -> tuple[int, int, int]:
    """Find indices of high-risk, low-risk, and borderline samples.

    Args:
        predictions: Amyloid probability predictions, shape [N].

    Returns:
        Tuple of (high_risk_idx, low_risk_idx, borderline_idx).
    """
    high_risk_idx = int(np.argmax(predictions))
    low_risk_idx = int(np.argmin(predictions))
    borderline_idx = int(np.argmin(np.abs(predictions - 0.5)))
    return high_risk_idx, low_risk_idx, borderline_idx


def main() -> None:
    """Entry point for SHAP analysis script."""
    parser = argparse.ArgumentParser(
        description="Run SHAP analysis on NeuroFusion-AD ADNI test set."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "models" / "final" / "best_model.pth"),
        help="Path to ADNI model checkpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use synthetic data (no checkpoint required).",
    )
    parser.add_argument(
        "--n-background",
        type=int,
        default=50,
        help="Number of background samples for KernelExplainer (default 50).",
    )
    parser.add_argument(
        "--n-explain",
        type=int,
        default=20,
        help="Number of test samples to explain (default 20).",
    )
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Determine whether to use checkpoint or synthetic data
    use_dry_run = args.dry_run or not Path(args.checkpoint).exists()
    if not args.dry_run and not Path(args.checkpoint).exists():
        log.warning(
            "Checkpoint not found; falling back to synthetic data",
            checkpoint=args.checkpoint,
        )

    # Load model — Phase 2B: embed_dim=256, num_heads=4 (remediated architecture)
    if not use_dry_run:
        ckpt_raw = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        ckpt_cfg = ckpt_raw.get("config", {})
        embed_dim = int(ckpt_cfg.get("embed_dim", 256))
        num_heads = int(ckpt_cfg.get("num_heads", 4))
        dropout = float(ckpt_cfg.get("dropout", 0.4))
        model = NeuroFusionAD(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        state = ckpt_raw.get("model_state_dict", ckpt_raw)
        model.load_state_dict(state)
        log.info("Loaded checkpoint", path=args.checkpoint, embed_dim=embed_dim, num_heads=num_heads)
    else:
        model = NeuroFusionAD(embed_dim=256, num_heads=4, dropout=0.4)
        log.info("Dry run: using randomly-initialised model weights")

    model.eval()

    from torch.utils.data import DataLoader

    # Phase 2B: use real ADNI CSVs (not synthetic) so feature dims match checkpoint
    adni_train_path = PROJECT_ROOT / "data" / "processed" / "adni" / "adni_train.csv"
    adni_test_path = PROJECT_ROOT / "data" / "processed" / "adni" / "adni_test.csv"

    if not use_dry_run and adni_train_path.exists() and adni_test_path.exists():
        log.info("Loading real ADNI datasets for SHAP")
        train_ds = NeuroFusionCSVDataset(
            str(adni_train_path), mode="adni", fit_imputation=True
        )
        test_ds = NeuroFusionCSVDataset(
            str(adni_test_path), mode="adni", fit_imputation=False,
            imputation_stats=train_ds.imputation_stats,
        )
        background_loader = DataLoader(train_ds, batch_size=args.n_background, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=args.n_explain, shuffle=False)
        background_batches = [next(iter(background_loader))]
        test_batches = [next(iter(test_loader))]
    else:
        log.info("Generating background dataset (synthetic fallback)", n=args.n_background + args.n_explain)
        full_dataset = generate_synthetic_adni(
            n_samples=args.n_background + args.n_explain, seed=0
        )
        from torch.utils.data import Subset
        background_dataset = Subset(full_dataset, list(range(args.n_background)))
        test_dataset = Subset(
            full_dataset,
            list(range(args.n_background, args.n_background + args.n_explain)),
        )
        background_loader = DataLoader(background_dataset, batch_size=args.n_background)
        test_loader = DataLoader(test_dataset, batch_size=args.n_explain)
        background_batches = list(background_loader)
        test_batches = list(test_loader)

    log.info(
        "Background/test data ready",
        n_background=args.n_background,
        n_test=args.n_explain,
    )

    # Initialise SHAP explainer
    log.info("Initialising KernelExplainer (this may take a moment)...")
    try:
        explainer = NeuralFusionSHAPExplainer(
            model=model,
            background_data=background_batches,
            device="cpu",
        )

        # Run explanation
        log.info("Computing SHAP values (may take several minutes)...")
        shap_results = explainer.explain(
            test_samples=test_batches,
            n_samples=args.n_explain,
        )

        # Save summary plot
        log.info("Saving SHAP summary plot")
        explainer.plot_summary(
            shap_results,
            str(FIGURES_DIR / "shap_summary.png"),
        )

        # Find interesting samples
        high_idx, low_idx, border_idx = _find_interesting_samples(
            shap_results["predictions"]
        )

        log.info(
            "Interesting sample indices",
            high_risk=high_idx,
            low_risk=low_idx,
            borderline=border_idx,
        )

        # Save waterfall plots
        for plot_idx, sample_idx in enumerate([high_idx, low_idx, border_idx]):
            explainer.plot_waterfall(
                shap_results,
                sample_idx=sample_idx,
                save_path=str(FIGURES_DIR / f"shap_waterfall_{plot_idx}.png"),
            )
            log.info(
                f"Waterfall plot {plot_idx} saved",
                sample_idx=sample_idx,
                probability=float(shap_results["predictions"][sample_idx]),
            )

        print("\nSHAP analysis complete.")
        print(f"Summary plot: {FIGURES_DIR / 'shap_summary.png'}")
        print(f"Waterfall 0 (high risk,  p={shap_results['predictions'][high_idx]:.3f}): "
              f"{FIGURES_DIR / 'shap_waterfall_0.png'}")
        print(f"Waterfall 1 (low risk,   p={shap_results['predictions'][low_idx]:.3f}): "
              f"{FIGURES_DIR / 'shap_waterfall_1.png'}")
        print(f"Waterfall 2 (borderline, p={shap_results['predictions'][border_idx]:.3f}): "
              f"{FIGURES_DIR / 'shap_waterfall_2.png'}")

    except ImportError:
        log.error(
            "shap library not installed. Run: pip install shap"
        )
        sys.exit(1)
    except Exception as exc:
        log.error("SHAP analysis failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
