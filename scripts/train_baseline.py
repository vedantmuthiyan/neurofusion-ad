#!/usr/bin/env python3
"""Train NeuroFusion-AD baseline on ADNI.

Loads the baseline training configuration, creates NeuroFusionCSVDataset instances
for ADNI train and val splits, fits the Bio-Hermes-001 digital scaler on the
BH train CSV (saves to data/processed/biohermes/biohermes_digital_scaler.pkl),
then trains the model using NeuroFusionTrainer.fit().

Usage:
    python scripts/train_baseline.py --config configs/training/baseline_config.yaml
    python scripts/train_baseline.py --config configs/training/baseline_config.yaml --resume

Exit codes:
    0: Training completed successfully with val_auc >= 0.74
    1: Training failed or val_auc < 0.74 (threshold for minimum acceptable performance)

IEC 62304 Requirement Traceability:
    SRS-001 § 5.5 — Training Requirements
    SAD-001 § 6.1 — Training Infrastructure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
import torch

import structlog

log = structlog.get_logger(__name__)

# Ensure project root is on the path
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
        Namespace with 'config' and 'resume' attributes.
    """
    parser = argparse.ArgumentParser(
        description="Train NeuroFusion-AD baseline on ADNI dataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/baseline_config.yaml",
        help="Path to baseline_config.yaml",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint in checkpoint_dir (if exists).",
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
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def fit_biohermes_scaler(cfg: dict) -> None:
    """Fit and save the Bio-Hermes-001 digital (acoustic+motor) scaler.

    Only fits if the scaler file does not already exist.

    Args:
        cfg: Full training config dict (expects 'data' sub-dict).
    """
    bh_train_path = Path(cfg["data"]["biohermes_train"])
    scaler_path = Path(cfg["data"]["biohermes_scaler_path"])

    if scaler_path.exists():
        log.info("BH digital scaler already exists — skipping fit", path=str(scaler_path))
        return

    if not bh_train_path.exists():
        log.warning(
            "Bio-Hermes-001 train CSV not found — skipping scaler fit",
            path=str(bh_train_path),
        )
        return

    log.info("Fitting Bio-Hermes-001 digital scaler", bh_train=str(bh_train_path))
    NeuroFusionCSVDataset.fit_biohermes_scaler(
        bh_train_csv_path=str(bh_train_path),
        save_path=str(scaler_path),
    )


def build_adni_loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    """Build ADNI train and val DataLoaders.

    Fits imputation statistics on the train CSV and applies them to val.

    Args:
        cfg: Full training config dict.

    Returns:
        Tuple of (train_loader, val_loader).

    Raises:
        FileNotFoundError: If ADNI CSV files are not found.
    """
    train_path = Path(cfg["data"]["adni_train"])
    val_path = Path(cfg["data"]["adni_val"])

    if not train_path.exists():
        raise FileNotFoundError(f"ADNI train CSV not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"ADNI val CSV not found: {val_path}")

    log.info("Building ADNI train dataset", path=str(train_path))
    train_ds = NeuroFusionCSVDataset(
        csv_path=str(train_path),
        mode="adni",
        fit_imputation=True,
    )

    log.info("Building ADNI val dataset", path=str(val_path))
    val_ds = NeuroFusionCSVDataset(
        csv_path=str(val_path),
        mode="adni",
        fit_imputation=False,
        imputation_stats=train_ds.imputation_stats,
    )

    batch_size = int(cfg["training"]["batch_size"])
    num_workers = 0  # Safe default (Windows + MacOS compatibility)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    log.info(
        "ADNI DataLoaders created",
        n_train=len(train_ds),
        n_val=len(val_ds),
        batch_size=batch_size,
    )
    return train_loader, val_loader


def build_model(cfg: dict) -> NeuroFusionAD:
    """Instantiate the NeuroFusionAD model from config.

    Args:
        cfg: Full training config dict with 'model' sub-dict.

    Returns:
        Initialized NeuroFusionAD model.
    """
    model_cfg = cfg.get("model", {})
    model = NeuroFusionAD(
        embed_dim=int(model_cfg.get("embed_dim", 768)),
        num_heads=int(model_cfg.get("num_heads", 8)),
        graph_threshold=float(model_cfg.get("graph_threshold", 0.7)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )
    n_params = model.count_parameters()
    log.info("Model built", total_trainable_params=n_params)
    return model


def main() -> int:
    """Main training entrypoint.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    args = parse_args()
    cfg = load_config(args.config)

    log.info("train_baseline starting", config=args.config)

    # 1. Fit Bio-Hermes digital scaler (before training, no-op if already exists)
    fit_biohermes_scaler(cfg)

    # 2. Build ADNI data loaders
    try:
        train_loader, val_loader = build_adni_loaders(cfg)
    except FileNotFoundError as e:
        log.error("Data loading failed", error=str(e))
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # 3. Build model
    model = build_model(cfg)

    # 4. Resume from checkpoint if requested
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.resume:
        ckpt_dir = Path(cfg["checkpoints"]["dir"])
        ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pth"))
        if ckpts:
            latest = ckpts[-1]
            log.info("Resuming from checkpoint", checkpoint=str(latest))
            try:
                NeuroFusionTrainer.load_checkpoint(model, str(latest), device=device)
            except Exception as exc:
                log.warning("Checkpoint load failed — starting from scratch", error=str(exc))
        else:
            log.info("No checkpoint found — starting fresh")

    # 5. Build loss function and trainer
    loss_weights = cfg.get("loss_weights", {"cls": 1.0, "reg": 0.5, "surv": 0.5})
    loss_fn = MultiTaskLoss(loss_weights)

    trainer_config = {
        **cfg["training"],
        "wandb": cfg.get("wandb", {}),
        "wandb_enabled": True,
    }
    trainer = NeuroFusionTrainer(model, config=trainer_config, device=device)

    # 6. Train
    n_epochs = int(cfg["training"]["n_epochs"])
    checkpoint_dir = cfg["checkpoints"]["dir"]
    save_every = int(cfg["checkpoints"].get("save_every_n_epochs", 10))

    results = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        n_epochs=n_epochs,
        checkpoint_dir=checkpoint_dir,
        save_every_n_epochs=save_every,
        run_name="adni_baseline",
    )

    best_val_auc = results["best_val_auc"]
    best_epoch = results["best_epoch"]

    print(f"\n{'='*60}")
    print(f"Training complete.")
    print(f"Best val AUC: {best_val_auc:.4f} (epoch {best_epoch})")
    print(f"{'='*60}")

    # 7. Check minimum performance threshold
    MIN_AUC_THRESHOLD = 0.74
    if best_val_auc < MIN_AUC_THRESHOLD:
        print(
            f"WARNING: val_auc {best_val_auc:.4f} below minimum threshold {MIN_AUC_THRESHOLD}",
            file=sys.stderr,
        )
        log.error(
            "training_below_threshold",
            best_val_auc=best_val_auc,
            threshold=MIN_AUC_THRESHOLD,
        )
        return 1

    log.info("train_baseline complete", best_val_auc=best_val_auc, best_epoch=best_epoch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
