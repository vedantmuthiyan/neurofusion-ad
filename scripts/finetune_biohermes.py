#!/usr/bin/env python3
"""Fine-tune NeuroFusion-AD on Bio-Hermes-001 with frozen modality encoders.

Loads the ADNI-trained model from models/final/best_model.pth, freezes the
four modality encoders (fluid, acoustic, motor, clinical), and trains only
the cross-modal attention, GNN, and classifier head on Bio-Hermes-001.

Bio-Hermes-001 is classification-only (amyloid positivity), so:
    loss_weights = {cls: 1.0, reg: 0.0, surv: 0.0}

At startup, prints which modules are frozen and verifies that frozen params
are less than 50% of total trainable parameters post-freeze.

Usage:
    python scripts/finetune_biohermes.py
    python scripts/finetune_biohermes.py --config configs/training/finetune_biohermes_config.yaml

Exit codes:
    0: Fine-tuning completed successfully with bh_val_auc >= 0.78
    1: Fine-tuning failed or bh_val_auc < 0.78

IEC 62304 Requirement Traceability:
    SRS-001 § 5.5 — Training Requirements (Domain Adaptation)
    SAD-001 § 6.3 — Bio-Hermes Fine-Tuning Strategy
    RMF-001 § 4.4 — Risk Control: Domain Shift Between Datasets
"""

from __future__ import annotations

import argparse
import pickle
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
        description="Fine-tune NeuroFusion-AD on Bio-Hermes-001 with frozen encoders."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/finetune_biohermes_config.yaml",
        help="Path to finetune_biohermes_config.yaml",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML fine-tuning configuration.

    Args:
        config_path: Path to finetune_biohermes_config.yaml.

    Returns:
        Config dict.

    Raises:
        FileNotFoundError: If config_path does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Fine-tune config not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def freeze_modules(model: NeuroFusionAD, module_names: list[str]) -> None:
    """Freeze specified named modules (set requires_grad=False).

    Args:
        model: NeuroFusionAD model to partially freeze.
        module_names: List of attribute names on the model to freeze
            (e.g., ['fluid_encoder', 'acoustic_encoder']).
    """
    print("\nFreezing modules:")
    for name in module_names:
        module = getattr(model, name, None)
        if module is None:
            log.warning("Freeze target not found on model", module_name=name)
            print(f"  WARNING: module '{name}' not found on model — skipping.")
            continue
        n_params = sum(p.numel() for p in module.parameters())
        for param in module.parameters():
            param.requires_grad = False
        print(f"  Frozen: {name} ({n_params:,} parameters)")
        log.info("module_frozen", name=name, n_params=n_params)


def verify_frozen_ratio(model: NeuroFusionAD) -> float:
    """Verify that more than 50% of parameters are still trainable after freezing.

    Prints a summary of trainable vs. frozen parameter counts.

    Args:
        model: Model after freezing.

    Returns:
        Fraction of parameters that are trainable (0.0 to 1.0).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    trainable_fraction = trainable_params / total_params if total_params > 0 else 0.0

    print(f"\nParameter summary:")
    print(f"  Total:     {total_params:>12,}")
    print(f"  Trainable: {trainable_params:>12,} ({trainable_fraction * 100:.1f}%)")
    print(f"  Frozen:    {frozen_params:>12,} ({(1 - trainable_fraction) * 100:.1f}%)")

    if trainable_fraction < 0.50:
        log.warning(
            "Less than 50% of parameters are trainable after freezing",
            trainable_fraction=trainable_fraction,
        )
        print(f"  WARNING: Only {trainable_fraction * 100:.1f}% of params are trainable. Expected > 50%.")
    else:
        print(f"  OK: {trainable_fraction * 100:.1f}% trainable — above 50% threshold.")

    return trainable_fraction


def load_biohermes_scaler(scaler_path: str) -> object | None:
    """Load the fitted Bio-Hermes-001 digital (acoustic+motor) scaler.

    Args:
        scaler_path: Path to biohermes_digital_scaler.pkl.

    Returns:
        Fitted StandardScaler, or None if file not found.
    """
    scaler_path = Path(scaler_path)
    if not scaler_path.exists():
        log.warning("BH digital scaler not found — raw features will be used", path=str(scaler_path))
        return None
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    log.info("BH digital scaler loaded", path=str(scaler_path))
    return scaler


def build_biohermes_loaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    """Build Bio-Hermes-001 train and val DataLoaders.

    Args:
        cfg: Full fine-tuning config dict.

    Returns:
        Tuple of (train_loader, val_loader).

    Raises:
        FileNotFoundError: If Bio-Hermes-001 CSV files are not found.
    """
    data_cfg = cfg.get("data", {})
    train_path = str(data_cfg.get("biohermes_train", "data/processed/biohermes/biohermes001_train.csv"))
    val_path = str(data_cfg.get("biohermes_val", "data/processed/biohermes/biohermes001_val.csv"))
    scaler_path = str(data_cfg.get("biohermes_scaler", "data/processed/biohermes/biohermes_digital_scaler.pkl"))

    if not Path(train_path).exists():
        raise FileNotFoundError(f"Bio-Hermes-001 train CSV not found: {train_path}")
    if not Path(val_path).exists():
        raise FileNotFoundError(f"Bio-Hermes-001 val CSV not found: {val_path}")

    bh_scaler = load_biohermes_scaler(scaler_path)

    train_ds = NeuroFusionCSVDataset(
        csv_path=train_path,
        mode="biohermes",
        fit_imputation=True,
        biohermes_scaler=bh_scaler,
    )
    val_ds = NeuroFusionCSVDataset(
        csv_path=val_path,
        mode="biohermes",
        fit_imputation=False,
        imputation_stats=train_ds.imputation_stats,
        biohermes_scaler=bh_scaler,
    )

    batch_size = int(cfg["training"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    log.info(
        "BH DataLoaders created",
        n_train=len(train_ds),
        n_val=len(val_ds),
        batch_size=batch_size,
    )
    return train_loader, val_loader


def main() -> int:
    """Main fine-tuning entrypoint.

    Returns:
        Exit code: 0 on success (bh_val_auc >= 0.78), 1 on failure.
    """
    args = parse_args()

    try:
        cfg = load_config(args.config)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    log.info("finetune_biohermes starting", config=args.config)

    # 1. Load pre-trained model
    pretrained_path = cfg.get("pretrained_checkpoint", "models/final/best_model.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = NeuroFusionAD(
        embed_dim=768,
        num_heads=8,
        graph_threshold=0.7,
        dropout=0.1,
    )

    if not Path(pretrained_path).exists():
        print(f"ERROR: Pretrained checkpoint not found: {pretrained_path}", file=sys.stderr)
        print("Run train_best_model.py first to generate models/final/best_model.pth", file=sys.stderr)
        log.error("pretrained_checkpoint_not_found", path=pretrained_path)
        return 1

    NeuroFusionTrainer.load_checkpoint(model, pretrained_path, device=device)
    model = model.to(device)
    print(f"\nLoaded pretrained checkpoint: {pretrained_path}")

    # 2. Freeze encoder modules
    frozen_modules = cfg.get("frozen_modules", [
        "fluid_encoder", "acoustic_encoder", "motor_encoder", "clinical_encoder"
    ])
    freeze_modules(model, frozen_modules)
    trainable_fraction = verify_frozen_ratio(model)

    # 3. Build Bio-Hermes-001 data loaders
    try:
        train_loader, val_loader = build_biohermes_loaders(cfg)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        log.error("Data loading failed", error=str(e))
        return 1

    # 4. Loss function: classification only
    loss_weights = cfg.get("loss_weights", {"cls": 1.0, "reg": 0.0, "surv": 0.0})
    assert float(loss_weights.get("reg", 0.0)) == 0.0, "reg_weight must be 0.0 for Bio-Hermes fine-tuning"
    assert float(loss_weights.get("surv", 0.0)) == 0.0, "surv_weight must be 0.0 for Bio-Hermes fine-tuning"
    loss_fn = MultiTaskLoss(loss_weights)
    print(f"\nLoss weights: cls={loss_weights['cls']}, reg={loss_weights['reg']}, surv={loss_weights['surv']}")

    # 5. Trainer (only optimizes requires_grad=True params)
    trainer_config = {
        **cfg["training"],
        "wandb_enabled": True,
        "wandb": cfg.get("wandb", {"project": "neurofusion-ad", "tags": ["finetune", "biohermes"]}),
    }
    trainer = NeuroFusionTrainer(model, config=trainer_config, device=device)

    # 6. Fine-tune
    checkpoint_dir = cfg["checkpoints"]["dir"]
    save_every = int(cfg["checkpoints"].get("save_every_n_epochs", 5))
    n_epochs = int(cfg["training"]["n_epochs"])

    results = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        n_epochs=n_epochs,
        checkpoint_dir=checkpoint_dir,
        save_every_n_epochs=save_every,
        run_name="biohermes_finetune",
    )

    bh_val_auc = results["best_val_auc"]
    best_epoch = results["best_epoch"]

    print(f"\n{'='*60}")
    print(f"Bio-Hermes fine-tuning complete.")
    print(f"Best BH val AUC: {bh_val_auc:.4f} (epoch {best_epoch})")
    print(f"{'='*60}")

    # 7. Performance threshold check
    MIN_AUC_THRESHOLD = 0.78
    if bh_val_auc < MIN_AUC_THRESHOLD:
        print(
            f"WARNING: BH val_auc {bh_val_auc:.4f} below minimum threshold {MIN_AUC_THRESHOLD}.",
            file=sys.stderr,
        )
        log.error(
            "biohermes_finetune_below_threshold",
            bh_val_auc=bh_val_auc,
            threshold=MIN_AUC_THRESHOLD,
        )
        return 1

    log.info(
        "finetune_biohermes complete",
        bh_val_auc=bh_val_auc,
        best_epoch=best_epoch,
        checkpoint_dir=checkpoint_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
