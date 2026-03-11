#!/usr/bin/env python3
"""Retrain NeuroFusion-AD with the best HPO configuration for 150 epochs.

Loads configs/training/best_config.yaml (produced by hpo_optuna.py),
trains for 150 epochs on ADNI, and saves the final model to
models/final/best_model.pth.

Usage:
    python scripts/train_best_model.py
    python scripts/train_best_model.py --config configs/training/best_config.yaml

Exit codes:
    0: Training completed successfully with val_auc >= 0.80
    1: Training failed or val_auc < 0.80

IEC 62304 Requirement Traceability:
    SRS-001 § 5.5 — Training Requirements
    SAD-001 § 6.1 — Training Infrastructure
    SRS-001 § 4.1 — Performance Requirements (AUC >= 0.85 target)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml
import torch

import structlog

log = structlog.get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.neurofusion_model import NeuroFusionAD
from src.data.csv_dataset import NeuroFusionCSVDataset
from src.training.losses import MultiTaskLoss
from src.training.trainer import NeuroFusionTrainer
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with 'config' attribute.
    """
    parser = argparse.ArgumentParser(
        description="Retrain NeuroFusion-AD with best HPO config for 150 epochs."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/best_config.yaml",
        help="Path to best_config.yaml (output of hpo_optuna.py).",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML training configuration.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Config dict.

    Raises:
        FileNotFoundError: If config_path does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Best config not found: {config_path}. "
            "Run hpo_optuna.py first to generate this file."
        )
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_adni_loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    """Build ADNI train and val DataLoaders.

    Args:
        cfg: Full training config dict.

    Returns:
        Tuple of (train_loader, val_loader).

    Raises:
        FileNotFoundError: If ADNI CSV files are not found.
    """
    data_cfg = cfg.get("data", {})
    train_path = str(data_cfg.get("adni_train", "data/processed/adni/adni_train.csv"))
    val_path = str(data_cfg.get("adni_val", "data/processed/adni/adni_val.csv"))

    if not Path(train_path).exists():
        raise FileNotFoundError(f"ADNI train CSV not found: {train_path}")
    if not Path(val_path).exists():
        raise FileNotFoundError(f"ADNI val CSV not found: {val_path}")

    train_ds = NeuroFusionCSVDataset(
        csv_path=train_path,
        mode="adni",
        fit_imputation=True,
    )
    val_ds = NeuroFusionCSVDataset(
        csv_path=val_path,
        mode="adni",
        fit_imputation=False,
        imputation_stats=train_ds.imputation_stats,
    )

    batch_size = int(cfg["training"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    log.info(
        "ADNI DataLoaders created",
        n_train=len(train_ds),
        n_val=len(val_ds),
        batch_size=batch_size,
    )
    return train_loader, val_loader


def main() -> int:
    """Main retraining entrypoint.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    args = parse_args()

    try:
        cfg = load_config(args.config)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    log.info("train_best_model starting", config=args.config)

    # Override epochs to 150 for full training
    cfg["training"]["n_epochs"] = 150
    cfg["training"]["early_stopping_patience"] = 25

    # Build data loaders
    try:
        train_loader, val_loader = build_adni_loaders(cfg)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        log.error("Data loading failed", error=str(e))
        return 1

    # Build model
    model_cfg = cfg.get("model", {})
    model = NeuroFusionAD(
        embed_dim=int(model_cfg.get("embed_dim", 768)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        graph_threshold=float(model_cfg.get("graph_threshold", 0.7)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
    log.info("Model built", n_params=model.count_parameters())

    # Loss function
    loss_weights = cfg.get("loss_weights", {"cls": 1.0, "reg": 0.5, "surv": 0.5})
    loss_fn = MultiTaskLoss(loss_weights)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer_config = {
        **cfg["training"],
        "wandb_enabled": True,
        "wandb": {"project": "neurofusion-ad", "tags": ["best_model", "150epochs"]},
    }
    trainer = NeuroFusionTrainer(model, config=trainer_config, device=device)

    # Train
    checkpoint_dir = "models/checkpoints/best_model_run"
    results = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        n_epochs=cfg["training"]["n_epochs"],
        checkpoint_dir=checkpoint_dir,
        save_every_n_epochs=10,
        run_name="best_model_150epochs",
    )

    best_val_auc = results["best_val_auc"]
    best_epoch = results["best_epoch"]

    print(f"\n{'='*60}")
    print(f"Full training complete.")
    print(f"Best val AUC: {best_val_auc:.4f} (epoch {best_epoch})")
    print(f"{'='*60}")

    # Copy best checkpoint to models/final/best_model.pth
    final_dir = Path("models/final")
    final_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = Path(checkpoint_dir) / "best_model.pth"
    final_path = final_dir / "best_model.pth"

    if best_ckpt.exists():
        shutil.copy2(str(best_ckpt), str(final_path))
        log.info("Final model saved", path=str(final_path))
        print(f"Final model saved to: {final_path}")
    else:
        log.warning("Best checkpoint not found", expected_path=str(best_ckpt))

    # Check performance threshold
    MIN_AUC_THRESHOLD = 0.80
    if best_val_auc < MIN_AUC_THRESHOLD:
        print(
            f"WARNING: val_auc {best_val_auc:.4f} below minimum threshold {MIN_AUC_THRESHOLD}. "
            "Consider more HPO trials or data quality improvements.",
            file=sys.stderr,
        )
        log.error(
            "best_model_below_threshold",
            best_val_auc=best_val_auc,
            threshold=MIN_AUC_THRESHOLD,
        )
        return 1

    log.info("train_best_model complete", best_val_auc=best_val_auc, final_path=str(final_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
