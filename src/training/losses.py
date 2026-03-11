"""Multi-task loss functions for NeuroFusion-AD training.

Implements masked multi-task loss combining classification (amyloid positivity),
regression (MMSE slope), and survival (Cox partial likelihood) objectives.
Supports missing labels via NaN masking so that incomplete data (e.g.,
Bio-Hermes-001 which has no MMSE slope or survival labels) can still be used.

IEC 62304 Requirement Traceability:
    SRS-001 § 5.5 — Training Objectives
    SAD-001 § 5.4 — Output Head Loss Functions
    RMF-001 § 4.2 — Risk Control: Guard against training on masked NaN labels
"""

from __future__ import annotations

import torch
import torch.nn as nn
import structlog

log = structlog.get_logger(__name__)


def cox_partial_likelihood_loss(
    log_hazard: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
) -> torch.Tensor:
    """Compute Cox partial likelihood loss using the Breslow approximation.

    Only computes the loss over samples where both time and event are not NaN.
    If no events are present in the batch (or no valid samples), returns a zero
    tensor so training can continue without crashing.

    The Breslow approximation computes:
        loss = -mean( log_h[event==1] - log( sum(exp(log_h[j])) for j in risk_set(i) ) )

    where the risk set for patient i is all patients j whose time_j >= time_i.

    Args:
        log_hazard: FloatTensor of shape [N, 1] or [N] — Cox log-hazard scores.
        time: FloatTensor of shape [N] or [N, 1] — observed/censored times.
        event: FloatTensor of shape [N] or [N, 1] — event indicator (1=event, 0=censored).

    Returns:
        Scalar tensor: mean negative Cox partial log-likelihood over events.
        Returns zero tensor if no events present in the valid mask.
    """
    log_hazard = log_hazard.squeeze(-1)  # [N]
    time = time.squeeze(-1)              # [N]
    event = event.squeeze(-1)            # [N]

    # Build valid mask: both time and event must be non-NaN
    valid_mask = ~(torch.isnan(time) | torch.isnan(event))

    if valid_mask.sum() == 0:
        log.warning("cox_partial_likelihood: no valid samples in batch — returning zero loss")
        return torch.tensor(0.0, device=log_hazard.device, requires_grad=True)

    lh_valid = log_hazard[valid_mask]   # [M]
    time_valid = time[valid_mask]       # [M]
    event_valid = event[valid_mask]     # [M]

    # Check if any events exist
    n_events = event_valid.sum()
    if n_events == 0:
        log.warning("cox_partial_likelihood: no events in batch — returning zero loss")
        return torch.tensor(0.0, device=log_hazard.device, requires_grad=True)

    # Sort by time descending (standard Cox convention)
    sort_order = torch.argsort(time_valid, descending=True)
    lh_sorted = lh_valid[sort_order]        # [M]
    event_sorted = event_valid[sort_order]  # [M]

    # Compute log of cumulative sum of exp(log_hazard) from the top (highest time)
    # This represents log(risk_set_sum) for each patient using Breslow approximation.
    # torch.cumsum on sorted-descending order gives sum of all patients with >= time.
    log_cumsum_h = torch.logcumsumexp(lh_sorted, dim=0)  # [M]

    # Cox partial likelihood (Breslow): only for patients who had an event
    event_mask = event_sorted == 1
    if event_mask.sum() == 0:
        return torch.tensor(0.0, device=log_hazard.device, requires_grad=True)

    log_partial = lh_sorted[event_mask] - log_cumsum_h[event_mask]  # [n_events]
    loss = -log_partial.mean()

    log.debug(
        "cox_partial_likelihood_loss",
        n_valid=int(valid_mask.sum()),
        n_events=int(n_events),
        loss=float(loss.detach()),
    )
    return loss


def augment_batch(batch: dict[str, torch.Tensor], noise_std: float = 0.01) -> dict[str, torch.Tensor]:
    """Add Gaussian noise to all continuous feature tensors in a batch.

    This augmentation is applied during training only to improve generalization.
    Label tensors (keys ending in '_label', 'slope', 'time', 'indicator') are
    never modified. Patient IDs are never modified.

    Args:
        batch: Dict containing feature tensors (e.g., 'fluid', 'acoustic', 'motor',
            'clinical') and label/metadata entries. Feature tensors receive noise;
            non-feature entries are passed through unchanged.
        noise_std: Standard deviation of Gaussian noise to add. Default 0.01.

    Returns:
        New dict with noisy copies of feature tensors; labels are identical to input.
    """
    feature_keys = {"fluid", "acoustic", "motor", "clinical"}
    augmented = {}
    for key, val in batch.items():
        if key in feature_keys and isinstance(val, torch.Tensor):
            noise = torch.randn_like(val) * noise_std
            augmented[key] = val + noise
        else:
            augmented[key] = val
    return augmented


class MultiTaskLoss(nn.Module):
    """Weighted multi-task loss with NaN-masked labels.

    Combines three loss components for NeuroFusion-AD training:
        1. Classification: BCEWithLogitsLoss on amyloid positivity (masked for NaN)
        2. Regression: MSELoss on MMSE slope (masked for NaN)
        3. Survival: Cox partial likelihood on time-to-event (masked for NaN/no-events)

    Loss weights control the relative contribution of each head. Setting a weight
    to 0.0 entirely skips that head (e.g., Bio-Hermes-001 fine-tuning uses
    loss_weights={'cls': 1.0, 'reg': 0.0, 'surv': 0.0}).

    Attributes:
        loss_weights: Dict with keys 'cls', 'reg', 'surv' and float values.
        bce_loss: BCEWithLogitsLoss (reduction='none' for masking).
        mse_loss: MSELoss (reduction='none' for masking).

    Args:
        loss_weights: Dict mapping 'cls', 'reg', 'surv' to float weights.
            Defaults to {'cls': 1.0, 'reg': 1.0, 'surv': 1.0}.

    Example:
        >>> loss_fn = MultiTaskLoss({'cls': 1.0, 'reg': 0.5, 'surv': 0.5})
        >>> preds = {
        ...     'amyloid_logit': torch.randn(8, 1),
        ...     'mmse_slope': torch.randn(8, 1),
        ...     'cox_log_hazard': torch.randn(8, 1),
        ... }
        >>> targets = {
        ...     'amyloid_label': torch.randint(0, 2, (8,)).float(),
        ...     'mmse_slope': torch.randn(8),
        ...     'survival_time': torch.rand(8) * 24,
        ...     'event_indicator': torch.randint(0, 2, (8,)).float(),
        ... }
        >>> result = loss_fn(preds, targets)
        >>> 'total' in result and 'cls' in result
        True
    """

    def __init__(self, loss_weights: dict[str, float] | None = None) -> None:
        """Initialize MultiTaskLoss with configurable per-head weights.

        Args:
            loss_weights: Dict with keys 'cls', 'reg', 'surv' and float weights.
                If None, defaults to all weights = 1.0.
        """
        super().__init__()
        defaults = {"cls": 1.0, "reg": 1.0, "surv": 1.0}
        if loss_weights is not None:
            defaults.update(loss_weights)
        self.loss_weights = defaults

        # reduction='none' allows per-sample masking before mean
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.mse_loss = nn.MSELoss(reduction="none")

        log.info(
            "MultiTaskLoss initialised",
            cls_weight=self.loss_weights["cls"],
            reg_weight=self.loss_weights["reg"],
            surv_weight=self.loss_weights["surv"],
        )

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute weighted multi-task loss with NaN masking.

        Args:
            predictions: Dict with keys:
                - 'amyloid_logit': FloatTensor [B, 1] — raw logit (pre-sigmoid)
                - 'mmse_slope': FloatTensor [B, 1] — predicted MMSE slope
                - 'cox_log_hazard': FloatTensor [B, 1] — Cox log-hazard
            targets: Dict with keys:
                - 'amyloid_label': FloatTensor [B] — binary label (NaN allowed)
                - 'mmse_slope': FloatTensor [B] — MMSE slope target (NaN allowed)
                - 'survival_time': FloatTensor [B] — event time (NaN allowed)
                - 'event_indicator': FloatTensor [B] — 0/1 event flag (NaN allowed)

        Returns:
            Dict with keys:
                - 'total': scalar tensor — weighted sum of active losses
                - 'cls': scalar tensor — classification loss (0 if weight=0 or all NaN)
                - 'reg': scalar tensor — regression loss (0 if weight=0 or all NaN)
                - 'surv': scalar tensor — survival loss (0 if weight=0 or no events)
        """
        device = predictions["amyloid_logit"].device
        zero = torch.tensor(0.0, device=device)

        # --- Classification loss (BCEWithLogitsLoss, NaN-masked) ---
        cls_loss = zero
        if self.loss_weights["cls"] > 0.0:
            amyloid_label = targets["amyloid_label"].squeeze(-1)  # [B]
            amyloid_logit = predictions["amyloid_logit"].squeeze(-1)  # [B]
            cls_mask = ~torch.isnan(amyloid_label)
            if cls_mask.sum() > 0:
                cls_raw = self.bce_loss(
                    amyloid_logit[cls_mask],
                    amyloid_label[cls_mask],
                )
                cls_loss = cls_raw.mean()
                log.debug(
                    "cls_loss_computed",
                    n_valid=int(cls_mask.sum()),
                    cls_loss=float(cls_loss.detach()),
                )
            else:
                log.warning("MultiTaskLoss: all amyloid_labels are NaN — cls_loss=0")

        # --- Regression loss (MSELoss, NaN-masked) ---
        reg_loss = zero
        if self.loss_weights["reg"] > 0.0:
            mmse_slope_target = targets["mmse_slope"].squeeze(-1)   # [B]
            mmse_slope_pred = predictions["mmse_slope"].squeeze(-1)  # [B]
            reg_mask = ~torch.isnan(mmse_slope_target)
            if reg_mask.sum() > 0:
                reg_raw = self.mse_loss(
                    mmse_slope_pred[reg_mask],
                    mmse_slope_target[reg_mask],
                )
                reg_loss = reg_raw.mean()
                log.debug(
                    "reg_loss_computed",
                    n_valid=int(reg_mask.sum()),
                    reg_loss=float(reg_loss.detach()),
                )
            else:
                log.warning("MultiTaskLoss: all mmse_slope labels are NaN — reg_loss=0")

        # --- Survival loss (Cox partial likelihood, NaN-masked via cox function) ---
        surv_loss = zero
        if self.loss_weights["surv"] > 0.0:
            surv_loss = cox_partial_likelihood_loss(
                log_hazard=predictions["cox_log_hazard"],
                time=targets["survival_time"],
                event=targets["event_indicator"],
            )

        # --- Weighted total ---
        total_loss = (
            self.loss_weights["cls"] * cls_loss
            + self.loss_weights["reg"] * reg_loss
            + self.loss_weights["surv"] * surv_loss
        )

        log.debug(
            "MultiTaskLoss forward",
            total=float(total_loss.detach()),
            cls=float(cls_loss.detach() if isinstance(cls_loss, torch.Tensor) else cls_loss),
            reg=float(reg_loss.detach() if isinstance(reg_loss, torch.Tensor) else reg_loss),
            surv=float(surv_loss.detach() if isinstance(surv_loss, torch.Tensor) else surv_loss),
        )

        return {
            "total": total_loss,
            "cls": cls_loss if isinstance(cls_loss, torch.Tensor) else torch.tensor(cls_loss, device=device),
            "reg": reg_loss if isinstance(reg_loss, torch.Tensor) else torch.tensor(reg_loss, device=device),
            "surv": surv_loss if isinstance(surv_loss, torch.Tensor) else torch.tensor(surv_loss, device=device),
        }
