"""Calibration evaluation and temperature scaling for NeuroFusion-AD.

Computes Expected Calibration Error (ECE) and provides post-hoc
temperature scaling to improve probability calibration.

Well-calibrated probabilities are required for clinical decision support —
a model that outputs 0.9 amyloid positivity probability should be positive
~90% of the time in the target population.

IEC 62304 compliance:
    - Temperature scaling is fit only on validation data (never test data).
    - ECE < 0.10 is a Phase 2 exit criterion (RMF-001 § 4.2).

Document traceability:
    SRS-001 § 6.2 — Calibration Requirements
    RMF-001 § 4.2 — Safety Monitoring: Probability Calibration
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import structlog

log = structlog.get_logger(__name__)


class CalibrationEvaluator:
    """Expected Calibration Error (ECE) computation and temperature scaling.

    ECE partitions predicted probabilities into n_bins equal-width bins and
    computes the weighted average of |accuracy - confidence| per bin.

    Temperature Scaling applies a single scalar T to the raw logits before
    sigmoid: P(y=1) = sigmoid(logit / T). T > 1 softens predictions (lower
    confidence), T < 1 sharpens them.

    Attributes:
        n_bins: Number of bins for ECE computation (default 10).

    Example:
        >>> import numpy as np
        >>> evaluator = CalibrationEvaluator()
        >>> probs = np.array([0.1, 0.4, 0.7, 0.9])
        >>> labels = np.array([0, 0, 1, 1])
        >>> ece = evaluator.compute_ece(probs, labels)
        >>> print(f"ECE: {ece:.4f}")
    """

    def __init__(self, n_bins: int = 10) -> None:
        """Initialise CalibrationEvaluator.

        Args:
            n_bins: Number of equally-spaced bins in [0, 1] for ECE (default 10).
        """
        self.n_bins = n_bins

    def compute_ece(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        n_bins: Optional[int] = None,
    ) -> float:
        """Compute Expected Calibration Error.

        Partitions predictions into n_bins equal-width bins. For each bin
        computes |mean_confidence - fraction_positive| weighted by bin size.

        Args:
            probabilities: Predicted probabilities in [0, 1], shape [N].
            labels: True binary labels {0, 1}, shape [N].
            n_bins: Number of bins (overrides instance default if provided).

        Returns:
            ECE value in [0, 1]. Lower is better. 0 = perfect calibration.

        Raises:
            ValueError: If probabilities and labels have different lengths.
        """
        probabilities = np.asarray(probabilities, dtype=float).ravel()
        labels = np.asarray(labels, dtype=float).ravel()

        if len(probabilities) != len(labels):
            raise ValueError(
                f"probabilities length ({len(probabilities)}) != "
                f"labels length ({len(labels)})"
            )

        n_bins_use = n_bins if n_bins is not None else self.n_bins
        n = len(probabilities)
        bin_boundaries = np.linspace(0.0, 1.0, n_bins_use + 1)

        ece = 0.0
        for i in range(n_bins_use):
            low = bin_boundaries[i]
            high = bin_boundaries[i + 1]
            # Include right edge in last bin
            if i == n_bins_use - 1:
                mask = (probabilities >= low) & (probabilities <= high)
            else:
                mask = (probabilities >= low) & (probabilities < high)

            n_bin = int(mask.sum())
            if n_bin == 0:
                continue

            conf = float(probabilities[mask].mean())
            acc = float(labels[mask].mean())
            ece += (n_bin / n) * abs(acc - conf)

        log.debug("ECE computed", ece=ece, n_samples=n, n_bins=n_bins_use)
        return float(ece)

    def fit_temperature(
        self,
        val_logits: np.ndarray,
        val_labels: np.ndarray,
    ) -> float:
        """Optimize temperature T on validation set using negative log-likelihood.

        Minimises the binary cross-entropy loss of sigmoid(logit / T) against
        val_labels over T in [0.1, 10.0].

        Args:
            val_logits: Raw logits (pre-sigmoid) from the model, shape [N].
            val_labels: True binary labels {0, 1}, shape [N].

        Returns:
            Optimal temperature T > 0. T = 1.0 means no scaling.
        """
        val_logits = np.asarray(val_logits, dtype=float).ravel()
        val_labels = np.asarray(val_labels, dtype=float).ravel()

        def nll(temperature: float) -> float:
            """Negative log-likelihood at temperature T."""
            scaled_probs = _sigmoid(val_logits / temperature)
            scaled_probs = np.clip(scaled_probs, 1e-7, 1 - 1e-7)
            loss = -np.mean(
                val_labels * np.log(scaled_probs)
                + (1 - val_labels) * np.log(1 - scaled_probs)
            )
            return float(loss)

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        optimal_t = float(result.x)

        log.info(
            "Temperature scaling complete",
            optimal_temperature=optimal_t,
            n_val=len(val_logits),
        )
        return optimal_t

    def apply_temperature(
        self,
        logits: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        """Apply temperature scaling to raw logits to produce calibrated probabilities.

        Args:
            logits: Raw model logits (pre-sigmoid), shape [N].
            temperature: Temperature parameter T > 0. T > 1 softens predictions.

        Returns:
            Calibrated probabilities in [0, 1], shape [N].

        Raises:
            ValueError: If temperature <= 0.
        """
        if temperature <= 0:
            raise ValueError(
                f"Temperature must be > 0, got {temperature}."
            )
        logits = np.asarray(logits, dtype=float).ravel()
        return _sigmoid(logits / temperature)

    def plot_reliability_diagram(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        save_path: str,
        title: str = "Reliability Diagram",
    ) -> None:
        """Save a reliability diagram (calibration plot).

        A perfectly calibrated model lies on the diagonal. Points above the
        diagonal indicate under-confidence; below indicates over-confidence.

        Args:
            probabilities: Predicted probabilities in [0, 1], shape [N].
            labels: True binary labels {0, 1}, shape [N].
            save_path: Absolute path to save the PNG figure.
            title: Plot title (default 'Reliability Diagram').
        """
        probabilities = np.asarray(probabilities, dtype=float).ravel()
        labels = np.asarray(labels, dtype=float).ravel()

        n_bins_use = self.n_bins
        bin_boundaries = np.linspace(0.0, 1.0, n_bins_use + 1)
        bin_midpoints = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

        mean_confs = []
        mean_accs = []

        for i in range(n_bins_use):
            low = bin_boundaries[i]
            high = bin_boundaries[i + 1]
            if i == n_bins_use - 1:
                mask = (probabilities >= low) & (probabilities <= high)
            else:
                mask = (probabilities >= low) & (probabilities < high)

            if mask.sum() == 0:
                mean_confs.append(bin_midpoints[i])
                mean_accs.append(float("nan"))
            else:
                mean_confs.append(float(probabilities[mask].mean()))
                mean_accs.append(float(labels[mask].mean()))

        mean_confs = np.array(mean_confs)
        mean_accs = np.array(mean_accs)
        valid = ~np.isnan(mean_accs)

        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Reliability diagram
        ax = axes[0]
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.plot(mean_confs[valid], mean_accs[valid], "o-", color="#1f77b4", label="Model")
        ax.set_xlabel("Mean Predicted Probability", fontsize=12)
        ax.set_ylabel("Fraction of Positives", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(alpha=0.3)

        # Histogram of predicted probabilities
        ax2 = axes[1]
        ax2.hist(probabilities, bins=n_bins_use, range=(0, 1), color="#ff7f0e", edgecolor="white")
        ax2.set_xlabel("Predicted Probability", fontsize=12)
        ax2.set_ylabel("Count", fontsize=12)
        ax2.set_title("Prediction Distribution", fontsize=14)
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        log.info("Reliability diagram saved", path=save_path)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function.

    Args:
        x: Input array.

    Returns:
        Sigmoid values clipped to (1e-7, 1 - 1e-7).
    """
    return np.clip(1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float))), 1e-7, 1 - 1e-7)
