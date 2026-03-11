"""NeuroFusion-AD training engine.

Provides NeuroFusionTrainer, which wraps the training loop with:
    - AMP (Automatic Mixed Precision) on CUDA devices
    - Gradient accumulation (default 4 steps)
    - OneCycleLR scheduler with configurable warm-up fraction
    - AdamW optimizer
    - Early stopping with patience
    - W&B logging for all metrics every epoch
    - Checkpoint saving (best model and periodic saves)
    - Per-epoch evaluation metrics: AUC, RMSE, C-index

Also exposes compute_metrics() for standalone evaluation.

IEC 62304 Requirement Traceability:
    SRS-001 § 5.5 — Training Requirements
    SAD-001 § 6.1 — Software Architecture: Training Infrastructure
    RMF-001 § 4.3 — Performance Monitoring
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

import structlog

log = structlog.get_logger(__name__)

# Optional imports — gracefully degraded if not available
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _WANDB_AVAILABLE = False
    log.warning("wandb not available — W&B logging disabled")

try:
    from sklearn.metrics import roc_auc_score
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False
    log.warning("scikit-learn not available — AUC/RMSE metrics disabled")

try:
    from lifelines.utils import concordance_index
    _LIFELINES_AVAILABLE = True
except ImportError:  # pragma: no cover
    _LIFELINES_AVAILABLE = False
    log.warning("lifelines not available — C-index metric disabled")

from src.training.losses import MultiTaskLoss, augment_batch


def compute_metrics(
    all_preds: dict[str, list[float]],
    all_targets: dict[str, list[float]],
) -> dict[str, float]:
    """Compute evaluation metrics from accumulated predictions and targets.

    Computes:
        - AUC: ROC AUC for amyloid classification (skips if < 2 classes or all NaN).
        - RMSE: Root mean squared error for MMSE slope (skips if all NaN).
        - C-index: Harrell's concordance index for survival (skips if all NaN).

    Args:
        all_preds: Dict with keys 'amyloid_prob', 'mmse_slope', 'cox_log_hazard'.
            Each value is a list of floats (possibly containing NaN).
        all_targets: Dict with keys 'amyloid_label', 'mmse_slope_target',
            'survival_time', 'event_indicator'.
            Each value is a list of floats (possibly containing NaN).

    Returns:
        Dict with keys 'auc', 'rmse', 'c_index'. Each value is a float or NaN
        if computation was not possible.
    """
    metrics: dict[str, float] = {"auc": float("nan"), "rmse": float("nan"), "c_index": float("nan")}

    import numpy as _np

    # --- AUC ---
    if _SKLEARN_AVAILABLE:
        try:
            y_true = _np.array(all_targets["amyloid_label"], dtype=_np.float32)
            y_score = _np.array(all_preds["amyloid_prob"], dtype=_np.float32)
            mask = ~_np.isnan(y_true) & ~_np.isnan(y_score)
            if mask.sum() > 0 and len(_np.unique(y_true[mask])) == 2:
                metrics["auc"] = float(roc_auc_score(y_true[mask], y_score[mask]))
        except Exception as exc:
            log.warning("auc_computation_failed", error=str(exc))

    # --- RMSE ---
    try:
        y_true_reg = _np.array(all_targets["mmse_slope_target"], dtype=_np.float32)
        y_pred_reg = _np.array(all_preds["mmse_slope"], dtype=_np.float32)
        mask_reg = ~_np.isnan(y_true_reg) & ~_np.isnan(y_pred_reg)
        if mask_reg.sum() > 0:
            mse = _np.mean((y_true_reg[mask_reg] - y_pred_reg[mask_reg]) ** 2)
            metrics["rmse"] = float(_np.sqrt(mse))
    except Exception as exc:
        log.warning("rmse_computation_failed", error=str(exc))

    # --- C-index ---
    if _LIFELINES_AVAILABLE:
        try:
            times = _np.array(all_targets["survival_time"], dtype=_np.float32)
            events = _np.array(all_targets["event_indicator"], dtype=_np.float32)
            risk = _np.array(all_preds["cox_log_hazard"], dtype=_np.float32)
            mask_surv = ~_np.isnan(times) & ~_np.isnan(events) & ~_np.isnan(risk)
            if mask_surv.sum() > 0 and events[mask_surv].sum() > 0:
                metrics["c_index"] = float(
                    concordance_index(times[mask_surv], -risk[mask_surv], events[mask_surv])
                )
        except Exception as exc:
            log.warning("c_index_computation_failed", error=str(exc))

    return metrics


class NeuroFusionTrainer:
    """Training engine for NeuroFusion-AD with AMP, grad accumulation, and W&B.

    Manages the full training lifecycle:
        1. train_epoch: one pass over training data with grad accumulation + AMP.
        2. evaluate: inference pass with metric computation (AUC, RMSE, C-index).
        3. fit: full training run with early stopping, checkpointing, and W&B logging.

    Attributes:
        model: NeuroFusionAD model.
        config: Training configuration dict.
        device: Target device string ('cuda' or 'cpu').
        optimizer: AdamW optimizer.
        scaler: GradScaler for AMP (None on CPU).

    Args:
        model: The NeuroFusionAD model to train.
        config: Training configuration dict with keys from baseline_config.yaml.
            Required keys: learning_rate, weight_decay, gradient_accumulation_steps,
            n_epochs, batch_size, early_stopping_patience, onecycle_pct_start.
        device: 'cuda' or 'cpu'. Defaults to 'cuda' if available, else 'cpu'.

    Example:
        >>> import torch
        >>> from src.models.neurofusion_model import NeuroFusionAD
        >>> model = NeuroFusionAD()
        >>> config = {'learning_rate': 3e-4, 'weight_decay': 1e-4,
        ...           'gradient_accumulation_steps': 1, 'n_epochs': 2,
        ...           'batch_size': 4, 'early_stopping_patience': 5,
        ...           'onecycle_pct_start': 0.3, 'augmentation_noise_std': 0.01}
        >>> trainer = NeuroFusionTrainer(model, config, device='cpu')
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict[str, Any],
        device: str | None = None,
    ) -> None:
        """Initialize NeuroFusionTrainer.

        Args:
            model: The NeuroFusionAD model (or any nn.Module compatible with the
                expected forward() signature).
            config: Training configuration dict.
            device: Device string. If None, auto-detects CUDA availability.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model.to(device)
        self.config = config

        lr = float(config.get("learning_rate", 3e-4))
        wd = float(config.get("weight_decay", 1e-4))
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=wd,
        )

        # AMP scaler — only active on CUDA
        self.use_amp = device == "cuda"
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        self.grad_accum_steps = int(config.get("gradient_accumulation_steps", 4))
        self.noise_std = float(config.get("augmentation_noise_std", 0.01))

        log.info(
            "NeuroFusionTrainer initialised",
            device=device,
            lr=lr,
            weight_decay=wd,
            grad_accum_steps=self.grad_accum_steps,
            use_amp=self.use_amp,
        )

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move all tensor values in a batch dict to the trainer device.

        Args:
            batch: Dict containing tensors and non-tensor values (e.g., patient_id).

        Returns:
            New dict with tensors moved to self.device; non-tensors unchanged.
        """
        moved = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(self.device)
            else:
                moved[k] = v
        return moved

    def _build_targets(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Extract target tensors from a batch dict.

        Args:
            batch: Batch dict from DataLoader (may include extra keys).

        Returns:
            Targets dict compatible with MultiTaskLoss.forward().
        """
        return {
            "amyloid_label": batch["amyloid_label"],
            "mmse_slope": batch["mmse_slope"],
            "survival_time": batch["survival_time"],
            "event_indicator": batch["event_indicator"],
        }

    def train_epoch(
        self,
        loader: DataLoader,
        loss_fn: MultiTaskLoss,
    ) -> dict[str, float]:
        """Run one training epoch with gradient accumulation and optional AMP.

        Args:
            loader: Training DataLoader.
            loss_fn: MultiTaskLoss instance.

        Returns:
            Dict with keys 'loss', 'cls_loss', 'reg_loss', 'surv_loss' —
            mean values over all batches in this epoch.
        """
        self.model.train()
        epoch_losses: dict[str, list[float]] = {
            "loss": [], "cls_loss": [], "reg_loss": [], "surv_loss": []
        }

        self.optimizer.zero_grad()
        accum_count = 0

        for batch_idx, batch in enumerate(loader):
            batch = self._move_batch(batch)

            # Augment feature tensors with Gaussian noise
            if self.noise_std > 0.0:
                batch = augment_batch(batch, noise_std=self.noise_std)

            targets = self._build_targets(batch)
            model_input = {k: batch[k] for k in ("fluid", "acoustic", "motor", "clinical")}

            # Forward pass (with AMP on CUDA)
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    predictions = self.model(model_input)
                    loss_dict = loss_fn(predictions, targets)
                    loss = loss_dict["total"] / self.grad_accum_steps
                self.scaler.scale(loss).backward()
            else:
                predictions = self.model(model_input)
                loss_dict = loss_fn(predictions, targets)
                loss = loss_dict["total"] / self.grad_accum_steps
                loss.backward()

            accum_count += 1

            if accum_count >= self.grad_accum_steps:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                accum_count = 0

                # Step scheduler if available (must be stepped per optimizer step)
                if hasattr(self, "scheduler") and self.scheduler is not None:
                    self.scheduler.step()

            epoch_losses["loss"].append(float(loss_dict["total"].detach()) )
            epoch_losses["cls_loss"].append(float(loss_dict["cls"].detach()))
            epoch_losses["reg_loss"].append(float(loss_dict["reg"].detach()))
            epoch_losses["surv_loss"].append(float(loss_dict["surv"].detach()))

        # Handle any remaining gradient accumulation steps at end of epoch
        if accum_count > 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            self.optimizer.zero_grad()

        mean_losses = {k: float(sum(v) / len(v)) if v else 0.0 for k, v in epoch_losses.items()}
        log.debug("train_epoch_complete", **mean_losses)
        return mean_losses

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        loss_fn: MultiTaskLoss,
    ) -> dict[str, float]:
        """Run inference on a DataLoader and compute evaluation metrics.

        Args:
            loader: Validation or test DataLoader.
            loss_fn: MultiTaskLoss instance.

        Returns:
            Dict with keys 'loss', 'auc', 'rmse', 'c_index'.
            Metric values are NaN if computation was not possible (e.g., all NaN targets).
        """
        self.model.eval()
        all_losses: list[float] = []
        all_preds: dict[str, list[float]] = {
            "amyloid_prob": [], "mmse_slope": [], "cox_log_hazard": []
        }
        all_targets: dict[str, list[float]] = {
            "amyloid_label": [], "mmse_slope_target": [],
            "survival_time": [], "event_indicator": []
        }

        for batch in loader:
            batch = self._move_batch(batch)
            targets = self._build_targets(batch)
            model_input = {k: batch[k] for k in ("fluid", "acoustic", "motor", "clinical")}

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    predictions = self.model(model_input)
                    loss_dict = loss_fn(predictions, targets)
            else:
                predictions = self.model(model_input)
                loss_dict = loss_fn(predictions, targets)

            all_losses.append(float(loss_dict["total"].detach()))

            # Collect predictions (apply sigmoid for classification probability)
            amyloid_prob = torch.sigmoid(predictions["amyloid_logit"]).squeeze(-1)
            all_preds["amyloid_prob"].extend(amyloid_prob.cpu().tolist())
            all_preds["mmse_slope"].extend(predictions["mmse_slope"].squeeze(-1).cpu().tolist())
            all_preds["cox_log_hazard"].extend(predictions["cox_log_hazard"].squeeze(-1).cpu().tolist())

            all_targets["amyloid_label"].extend(targets["amyloid_label"].cpu().tolist())
            all_targets["mmse_slope_target"].extend(targets["mmse_slope"].cpu().tolist())
            all_targets["survival_time"].extend(targets["survival_time"].cpu().tolist())
            all_targets["event_indicator"].extend(targets["event_indicator"].cpu().tolist())

        mean_loss = sum(all_losses) / len(all_losses) if all_losses else float("nan")
        metrics = compute_metrics(all_preds, all_targets)
        result = {"loss": mean_loss, **metrics}
        log.debug("evaluate_complete", **{k: round(v, 4) if not math.isnan(v) else v for k, v in result.items()})
        return result

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: MultiTaskLoss,
        n_epochs: int,
        checkpoint_dir: str,
        save_every_n_epochs: int = 10,
        run_name: str | None = None,
    ) -> dict[str, Any]:
        """Run the full training loop with early stopping and checkpointing.

        Sets up OneCycleLR scheduler, then iterates epochs. Saves the best
        checkpoint (by val AUC). Logs all metrics to W&B if available.
        Stops early if val_auc does not improve for patience epochs.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            loss_fn: MultiTaskLoss instance.
            n_epochs: Maximum number of training epochs.
            checkpoint_dir: Directory path to save checkpoints.
            save_every_n_epochs: Save a periodic checkpoint every N epochs.
            run_name: Optional W&B run name.

        Returns:
            Dict with keys:
                - 'best_val_auc': float — best validation AUC achieved.
                - 'best_epoch': int — epoch number where best AUC was achieved.
                - 'history': list of dicts — per-epoch metric records.
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        patience = int(self.config.get("early_stopping_patience", 25))
        pct_start = float(self.config.get("onecycle_pct_start", 0.3))
        lr = float(self.config.get("learning_rate", 3e-4))

        # Total optimizer steps = epochs * (batches_per_epoch / grad_accum)
        steps_per_epoch = max(1, len(train_loader) // self.grad_accum_steps)
        total_steps = n_epochs * steps_per_epoch

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy="cos",
        )

        # Initialize W&B run if available
        wandb_enabled = _WANDB_AVAILABLE and self.config.get("wandb_enabled", True)
        if wandb_enabled:
            try:
                wandb_cfg = self.config.get("wandb", {})
                wandb.init(
                    project=wandb_cfg.get("project", "neurofusion-ad"),
                    entity=wandb_cfg.get("entity") or None,
                    name=run_name,
                    config=self.config,
                    tags=wandb_cfg.get("tags", []),
                    reinit=True,
                )
                log.info("wandb_run_initialised")
            except Exception as exc:
                log.warning("wandb_init_failed", error=str(exc))
                wandb_enabled = False

        history: list[dict[str, Any]] = []
        best_val_auc = float("-inf")
        best_epoch = 0
        epochs_no_improve = 0

        log.info("fit_start", n_epochs=n_epochs, patience=patience, device=self.device)

        for epoch in range(1, n_epochs + 1):
            t_start = time.time()

            train_metrics = self.train_epoch(train_loader, loss_fn)
            val_metrics = self.evaluate(val_loader, loss_fn)

            elapsed = time.time() - t_start
            val_auc = val_metrics.get("auc", float("nan"))

            epoch_record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_cls_loss": train_metrics["cls_loss"],
                "train_reg_loss": train_metrics["reg_loss"],
                "train_surv_loss": train_metrics["surv_loss"],
                "val_loss": val_metrics["loss"],
                "val_auc": val_auc,
                "val_rmse": val_metrics.get("rmse", float("nan")),
                "val_c_index": val_metrics.get("c_index", float("nan")),
                "elapsed_s": elapsed,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            history.append(epoch_record)

            log.info(
                "epoch_complete",
                epoch=epoch,
                train_loss=round(train_metrics["loss"], 4),
                val_loss=round(val_metrics["loss"], 4),
                val_auc=round(val_auc, 4) if not math.isnan(val_auc) else val_auc,
                elapsed_s=round(elapsed, 1),
            )

            # W&B logging
            if wandb_enabled:
                try:
                    wandb.log(epoch_record, step=epoch)
                except Exception as exc:
                    log.warning("wandb_log_failed", error=str(exc))

            # Periodic checkpoint
            if epoch % save_every_n_epochs == 0:
                ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pth"
                self._save_checkpoint(ckpt_path, epoch, epoch_record)

            # Best model checkpoint (by val_auc)
            auc_improved = (
                not math.isnan(val_auc) and val_auc > best_val_auc
            )
            if auc_improved:
                best_val_auc = val_auc
                best_epoch = epoch
                epochs_no_improve = 0
                best_path = checkpoint_dir / "best_model.pth"
                self._save_checkpoint(best_path, epoch, epoch_record)
                log.info("new_best_model", epoch=epoch, val_auc=round(best_val_auc, 4))
            else:
                # Only count toward patience when AUC is computable
                if not math.isnan(val_auc):
                    epochs_no_improve += 1

            # Early stopping
            if epochs_no_improve >= patience:
                log.info(
                    "early_stopping_triggered",
                    epoch=epoch,
                    patience=patience,
                    best_epoch=best_epoch,
                    best_val_auc=round(best_val_auc, 4),
                )
                break

        if wandb_enabled:
            try:
                wandb.finish()
            except Exception:
                pass

        log.info(
            "fit_complete",
            best_val_auc=best_val_auc,
            best_epoch=best_epoch,
            total_epochs_run=len(history),
        )
        return {
            "best_val_auc": best_val_auc,
            "best_epoch": best_epoch,
            "history": history,
        }

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: dict[str, Any],
    ) -> None:
        """Save model and optimizer state to a checkpoint file.

        Args:
            path: Destination file path for the checkpoint.
            epoch: Current epoch number (stored in checkpoint metadata).
            metrics: Metrics dict for this epoch (stored in checkpoint metadata).
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }
        torch.save(checkpoint, str(path))
        log.debug("checkpoint_saved", path=str(path), epoch=epoch)

    @staticmethod
    def load_checkpoint(
        model: nn.Module,
        checkpoint_path: str,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Load model weights from a checkpoint file.

        Args:
            model: Model instance to load weights into (in-place).
            checkpoint_path: Path to the .pth checkpoint file.
            device: Device to map tensors to during loading.

        Returns:
            Full checkpoint dict (includes 'epoch', 'metrics', 'config').

        Raises:
            FileNotFoundError: If checkpoint_path does not exist.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        log.info(
            "checkpoint_loaded",
            path=str(checkpoint_path),
            epoch=checkpoint.get("epoch"),
        )
        return checkpoint
