#!/usr/bin/env python3
"""Hyperparameter optimization for NeuroFusion-AD using Optuna.

Runs 30 trials of 40 epochs each with Optuna's MedianPruner.
Uses SQLite persistent storage so interrupted studies can be resumed automatically.
Saves the best configuration to configs/training/best_config.yaml when complete.

The objective function:
    - Samples hyperparameters from hpo_config.yaml search_space
    - Builds model, loss, trainer, and ADNI data loaders per trial
    - Reports val_auc after each epoch (enables Optuna pruning)
    - Returns best val_auc across all epochs in the trial

Usage:
    python scripts/hpo_optuna.py --config configs/training/hpo_config.yaml
    python scripts/hpo_optuna.py --config configs/training/hpo_config.yaml --n-trials 10

IEC 62304 Requirement Traceability:
    SRS-001 § 5.5 — Training Requirements (Hyperparameter Selection)
    SAD-001 § 6.2 — HPO Strategy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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

try:
    import optuna
    from optuna.pruners import MedianPruner
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
    log.error("optuna not installed — run: pip install optuna")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with 'config' and 'n_trials' attributes.
    """
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for NeuroFusion-AD."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/hpo_config.yaml",
        help="Path to hpo_config.yaml",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override n_trials from config (for quick testing).",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML HPO configuration.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Config dict.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"HPO config not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_loaders_from_fixed(fixed: dict, batch_size: int) -> tuple[DataLoader, DataLoader]:
    """Build ADNI train and val DataLoaders from fixed config section.

    Args:
        fixed: 'fixed' sub-dict from hpo_config.yaml with data paths.
        batch_size: Batch size for this trial.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    data_cfg = fixed.get("data", {})
    train_path = str(data_cfg.get("adni_train", "data/processed/adni/adni_train.csv"))
    val_path = str(data_cfg.get("adni_val", "data/processed/adni/adni_val.csv"))

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
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def sample_hyperparameters(trial: "optuna.Trial", search_space: dict) -> dict:
    """Sample hyperparameters for a trial from the search space definition.

    Args:
        trial: Optuna Trial object for suggesting values.
        search_space: Dict mapping param name to type/bounds/choices.

    Returns:
        Dict mapping param name to sampled value.
    """
    params = {}
    for name, spec in search_space.items():
        param_type = spec.get("type", "uniform")
        if param_type == "log_uniform":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
        elif param_type == "uniform":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"])
        elif param_type == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif param_type == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        else:
            log.warning("Unknown HPO param type", name=name, type=param_type)
    return params


def make_objective(hpo_cfg: dict) -> callable:
    """Create the Optuna objective function from the HPO configuration.

    Args:
        hpo_cfg: Full HPO config dict.

    Returns:
        Callable objective(trial) -> float (best val_auc for the trial).
    """
    search_space = hpo_cfg.get("search_space", {})
    fixed = hpo_cfg.get("fixed", {})
    n_epochs = int(hpo_cfg.get("n_epochs_per_trial", 40))
    patience = int(hpo_cfg.get("early_stopping_patience", 15))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def objective(trial: "optuna.Trial") -> float:
        """Objective function for a single Optuna trial.

        Args:
            trial: Optuna Trial instance for sampling hyperparameters.

        Returns:
            Best val_auc achieved in this trial (maximized).

        Raises:
            optuna.TrialPruned: If Optuna MedianPruner prunes this trial.
        """
        params = sample_hyperparameters(trial, search_space)

        batch_size = int(params.get("batch_size", 32))
        grad_accum = int(params.get("gradient_accumulation_steps", 4))
        lr = float(params.get("learning_rate", 3e-4))
        wd = float(params.get("weight_decay", 1e-4))
        dropout = float(params.get("dropout", 0.1))
        graph_threshold = float(params.get("graph_threshold", 0.7))
        pct_start = float(params.get("onecycle_pct_start", 0.3))
        cls_w = float(params.get("cls_weight", 1.0))
        reg_w = float(params.get("reg_weight", 0.5))
        surv_w = float(params.get("surv_weight", 0.5))

        try:
            train_loader, val_loader = build_loaders_from_fixed(fixed, batch_size)
        except FileNotFoundError as e:
            log.error("Data loading failed in HPO trial", error=str(e))
            return float("nan")

        model = NeuroFusionAD(
            embed_dim=int(fixed.get("embed_dim", 768)),
            num_heads=int(fixed.get("num_heads", 8)),
            graph_threshold=graph_threshold,
            dropout=dropout,
        )

        loss_fn = MultiTaskLoss({"cls": cls_w, "reg": reg_w, "surv": surv_w})

        trainer_config = {
            "learning_rate": lr,
            "weight_decay": wd,
            "gradient_accumulation_steps": grad_accum,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "early_stopping_patience": patience,
            "onecycle_pct_start": pct_start,
            "augmentation_noise_std": 0.01,
            "wandb_enabled": False,  # Disable W&B inside HPO trials
        }
        trainer = NeuroFusionTrainer(model, config=trainer_config, device=device)

        best_trial_auc = float("-inf")
        import math

        for epoch in range(1, n_epochs + 1):
            trainer.train_epoch(train_loader, loss_fn)
            val_metrics = trainer.evaluate(val_loader, loss_fn)
            val_auc = val_metrics.get("auc", float("nan"))

            if not math.isnan(val_auc):
                if val_auc > best_trial_auc:
                    best_trial_auc = val_auc
                trial.report(val_auc, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_trial_auc if best_trial_auc > float("-inf") else float("nan")

    return objective


def save_best_config(study: "optuna.Study", hpo_cfg: dict, output_path: str) -> None:
    """Save the best trial's hyperparameters as a training config YAML.

    Args:
        study: Completed Optuna study.
        hpo_cfg: HPO config dict (for fixed parameters).
        output_path: Path for the output best_config.yaml file.
    """
    best = study.best_trial
    fixed = hpo_cfg.get("fixed", {})
    params = best.params

    best_config = {
        "model": {
            "embed_dim": int(fixed.get("embed_dim", 768)),
            "num_heads": int(fixed.get("num_heads", 8)),
            "gnn_layers": int(fixed.get("gnn_layers", 3)),
            "dropout": float(params.get("dropout", 0.1)),
            "graph_threshold": float(params.get("graph_threshold", 0.7)),
        },
        "training": {
            "n_epochs": 150,  # Full training uses 150 epochs
            "batch_size": int(params.get("batch_size", 32)),
            "gradient_accumulation_steps": int(params.get("gradient_accumulation_steps", 4)),
            "learning_rate": float(params.get("learning_rate", 3e-4)),
            "weight_decay": float(params.get("weight_decay", 1e-4)),
            "early_stopping_patience": 25,
            "onecycle_pct_start": float(params.get("onecycle_pct_start", 0.3)),
            "augmentation_noise_std": 0.01,
        },
        "loss_weights": {
            "cls": float(params.get("cls_weight", 1.0)),
            "reg": float(params.get("reg_weight", 0.5)),
            "surv": float(params.get("surv_weight", 0.5)),
        },
        "data": fixed.get("data", {}),
        "hpo_metadata": {
            "best_trial_number": best.number,
            "best_val_auc": float(best.value),
            "n_trials_completed": len(study.trials),
            "study_name": study.study_name,
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)

    log.info("Best config saved", path=str(output_path), best_val_auc=float(best.value))


def main() -> int:
    """Main HPO entrypoint.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    if not _OPTUNA_AVAILABLE:
        print("ERROR: optuna is not installed. Run: pip install optuna", file=sys.stderr)
        return 1

    args = parse_args()
    hpo_cfg = load_config(args.config)

    study_name = hpo_cfg.get("study_name", "neurofusion_hpo")
    storage = hpo_cfg.get("storage", "sqlite:///optuna_study.db")
    n_trials = args.n_trials or int(hpo_cfg.get("n_trials", 30))
    direction = hpo_cfg.get("direction", "maximize")

    pruner_n_startup = int(hpo_cfg.get("pruner_n_startup_trials", 5))
    pruner_n_warmup = int(hpo_cfg.get("pruner_n_warmup_steps", 10))

    pruner = MedianPruner(
        n_startup_trials=pruner_n_startup,
        n_warmup_steps=pruner_n_warmup,
    )

    log.info(
        "HPO study starting",
        study_name=study_name,
        storage=storage,
        n_trials=n_trials,
        direction=direction,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        pruner=pruner,
        load_if_exists=True,  # Resume if study already exists
    )

    objective = make_objective(hpo_cfg)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Print top 5 trials
    print("\n" + "=" * 60)
    print("HPO COMPLETE — Top 5 Trials by val_auc:")
    print("=" * 60)
    completed = [t for t in study.trials if t.value is not None]
    completed_sorted = sorted(completed, key=lambda t: t.value if t.value is not None else -1, reverse=True)
    for i, trial in enumerate(completed_sorted[:5], 1):
        print(f"  #{i} Trial {trial.number}: val_auc={trial.value:.4f}")
        for k, v in trial.params.items():
            print(f"      {k}: {v}")
    print("=" * 60)

    best_config_path = "configs/training/best_config.yaml"
    try:
        save_best_config(study, hpo_cfg, best_config_path)
        print(f"\nBest config saved to: {best_config_path}")
    except Exception as exc:
        log.error("Failed to save best config", error=str(exc))

    print(f"Best val_auc: {study.best_value:.4f}")
    log.info("hpo_complete", best_val_auc=study.best_value, n_trials_completed=len(study.trials))
    return 0


if __name__ == "__main__":
    sys.exit(main())
