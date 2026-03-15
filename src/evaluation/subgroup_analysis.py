"""Subgroup analysis for NeuroFusion-AD fairness and equity evaluation.

Evaluates model performance across demographic subgroups (age, sex, APOE4
carrier status) to identify and quantify performance disparities.

Regulatory requirement: Maximum AUC gap across all subgroup pairs must be
< 0.07 for Phase 2 exit (RMF-001 § 5.3 — Equity and Bias Assessment).

A fairness_pass = True does NOT guarantee absence of bias; it indicates the
model meets the minimum equity threshold for clinical deployment.

IEC 62304 compliance:
    - No PHI is logged; results are aggregated metrics only.
    - All subgroup definitions are fixed and documented here.

Document traceability:
    SRS-001 § 6.3 — Equity and Subgroup Performance Requirements
    RMF-001 § 5.3 — Bias and Fairness Risk Controls
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import structlog
from sklearn.metrics import roc_auc_score

log = structlog.get_logger(__name__)

# Maximum allowable AUC gap across subgroup pairs (RMF-001 § 5.3)
MAX_AUC_GAP_THRESHOLD = 0.07

# Minimum patients required in a subgroup to compute AUC
MIN_SUBGROUP_SIZE = 5


class SubgroupAnalyzer:
    """Evaluate model performance across demographic subgroups.

    Subgroups evaluated:
        - Age: <65, 65-75, >75 (years)
        - Sex: Male (SEX_CODE=1), Female (SEX_CODE=0)
        - APOE4: Non-carrier (APOE4_COUNT=0), Carrier (APOE4_COUNT>=1)

    For each subgroup, AUC is computed with 95% bootstrap CI. The maximum
    AUC gap across all subgroup pairs is computed and compared to the
    regulatory threshold of 0.07.

    Attributes:
        n_bootstrap: Number of bootstrap iterations for CI (default 500).
        random_state: Seed for reproducibility.

    Example:
        >>> analyzer = SubgroupAnalyzer()
        >>> results = analyzer.analyze(model_outputs, targets, metadata_df)
        >>> print(results['fairness_pass'])
    """

    def __init__(
        self,
        n_bootstrap: int = 500,
        random_state: int = 42,
    ) -> None:
        """Initialise SubgroupAnalyzer.

        Args:
            n_bootstrap: Bootstrap iterations for CI computation (default 500).
            random_state: Random seed for reproducible bootstrap sampling.
        """
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

    def analyze(
        self,
        model_outputs: dict,
        targets: dict,
        metadata: pd.DataFrame,
    ) -> dict:
        """Run full subgroup analysis.

        Computes AUC with 95% CI for each demographic subgroup. Subgroups
        with fewer than MIN_SUBGROUP_SIZE patients return NaN AUC (no crash).

        Args:
            model_outputs: Model output dict with key 'amyloid_logit',
                shape [N, 1] or [N].
            targets: Ground truth dict with key 'amyloid_label', shape [N].
            metadata: DataFrame with columns AGE, SEX_CODE, APOE4_COUNT.
                Must have the same length N as model_outputs.

        Returns:
            Dict with subgroup AUC results:
                {
                    'age_lt65': {'auc': float, 'n': int, 'ci': tuple},
                    'age_65_75': {'auc': float, 'n': int, 'ci': tuple},
                    'age_gt75': {'auc': float, 'n': int, 'ci': tuple},
                    'sex_male': {'auc': float, 'n': int, 'ci': tuple},
                    'sex_female': {'auc': float, 'n': int, 'ci': tuple},
                    'apoe_noncarrier': {'auc': float, 'n': int, 'ci': tuple},
                    'apoe_carrier': {'auc': float, 'n': int, 'ci': tuple},
                    'max_auc_gap': float,
                    'fairness_pass': bool,
                }

        Raises:
            KeyError: If required columns are missing from metadata.
            ValueError: If array lengths are inconsistent.
        """
        # Validate metadata columns
        required_cols = {"AGE", "SEX_CODE", "APOE4_COUNT"}
        missing = required_cols - set(metadata.columns)
        if missing:
            raise KeyError(f"metadata is missing columns: {missing}")

        logits = np.asarray(model_outputs["amyloid_logit"]).ravel()
        amyloid_true = np.asarray(targets["amyloid_label"]).ravel()
        probs = _sigmoid(logits)

        n = len(probs)
        if len(amyloid_true) != n:
            raise ValueError(
                f"Inconsistent lengths: probs={n}, labels={len(amyloid_true)}"
            )
        if len(metadata) != n:
            raise ValueError(
                f"Inconsistent lengths: probs={n}, metadata={len(metadata)}"
            )

        age = metadata["AGE"].values.astype(float)
        sex = metadata["SEX_CODE"].values.astype(float)
        apoe4 = metadata["APOE4_COUNT"].values.astype(float)

        # Define subgroup masks
        subgroups = {
            "age_lt65": age < 65,
            "age_65_75": (age >= 65) & (age <= 75),
            "age_gt75": age > 75,
            "sex_male": sex == 1,
            "sex_female": sex == 0,
            "apoe_noncarrier": apoe4 == 0,
            "apoe_carrier": apoe4 >= 1,
        }

        results: dict = {}
        valid_aucs = []

        for name, mask in subgroups.items():
            mask = np.asarray(mask, dtype=bool)
            n_sub = int(mask.sum())
            if n_sub < MIN_SUBGROUP_SIZE:
                log.warning(
                    "SubgroupAnalyzer: subgroup too small",
                    subgroup=name,
                    n=n_sub,
                    min_required=MIN_SUBGROUP_SIZE,
                )
                results[name] = {
                    "auc": float("nan"),
                    "n": n_sub,
                    "ci": (float("nan"), float("nan")),
                }
                continue

            y_true_sub = amyloid_true[mask]
            y_prob_sub = probs[mask]

            # Filter NaN labels before AUC computation (ADNI has ~36% NaN amyloid labels)
            _valid = ~np.isnan(y_true_sub)
            y_true_sub = y_true_sub[_valid]
            y_prob_sub = y_prob_sub[_valid]

            if len(y_true_sub) < MIN_SUBGROUP_SIZE:
                log.warning(
                    "SubgroupAnalyzer: subgroup too small after NaN filter",
                    subgroup=name,
                    n_valid=int(len(y_true_sub)),
                )
                results[name] = {
                    "auc": float("nan"),
                    "n": n_sub,
                    "ci": (float("nan"), float("nan")),
                }
                continue

            # Check that we have both classes in subgroup
            if len(np.unique(y_true_sub)) < 2:
                log.warning(
                    "SubgroupAnalyzer: subgroup has only one class",
                    subgroup=name,
                    n=n_sub,
                )
                results[name] = {
                    "auc": float("nan"),
                    "n": n_sub,
                    "ci": (float("nan"), float("nan")),
                }
                continue

            try:
                auc = float(roc_auc_score(y_true_sub, y_prob_sub))
                ci = self._bootstrap_auc(y_true_sub, y_prob_sub)
                results[name] = {"auc": auc, "n": n_sub, "ci": ci}
                valid_aucs.append(auc)
                log.info(
                    "SubgroupAnalyzer: subgroup AUC computed",
                    subgroup=name,
                    auc=auc,
                    n=n_sub,
                )
            except Exception as exc:
                log.warning(
                    "SubgroupAnalyzer: AUC computation failed",
                    subgroup=name,
                    error=str(exc),
                )
                results[name] = {
                    "auc": float("nan"),
                    "n": n_sub,
                    "ci": (float("nan"), float("nan")),
                }

        # Compute max AUC gap
        if len(valid_aucs) >= 2:
            max_gap = float(max(valid_aucs) - min(valid_aucs))
        else:
            max_gap = float("nan")

        fairness_pass = (
            (not np.isnan(max_gap)) and (max_gap < MAX_AUC_GAP_THRESHOLD)
        )

        results["max_auc_gap"] = max_gap
        results["fairness_pass"] = fairness_pass

        log.info(
            "SubgroupAnalyzer complete",
            max_auc_gap=max_gap,
            fairness_pass=fairness_pass,
            n_subgroups_valid=len(valid_aucs),
        )
        return results

    def _bootstrap_auc(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> tuple[float, float]:
        """Compute 95% bootstrap CI for AUC.

        Args:
            y_true: Binary ground truth labels.
            y_prob: Predicted probabilities.

        Returns:
            Tuple (lower_ci, upper_ci) at 2.5th and 97.5th percentiles.
        """
        n = len(y_true)
        boot_scores = []
        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            yt = y_true[idx]
            yp = y_prob[idx]
            try:
                if len(np.unique(yt)) < 2:
                    continue
                with np.errstate(invalid="ignore"):
                    score = float(roc_auc_score(yt, yp))
                boot_scores.append(score)
            except Exception:
                continue

        if len(boot_scores) < 10:
            return (float("nan"), float("nan"))

        lower = float(np.percentile(boot_scores, 2.5))
        upper = float(np.percentile(boot_scores, 97.5))
        return (lower, upper)

    def plot_subgroup_comparison(
        self,
        results: dict,
        save_path: str,
    ) -> None:
        """Save bar chart of AUC by subgroup with 95% confidence intervals.

        Args:
            results: Dict returned by analyze(). Must contain subgroup keys
                and 'max_auc_gap'.
            save_path: Absolute path to save the PNG figure.
        """
        subgroup_keys = [
            "age_lt65", "age_65_75", "age_gt75",
            "sex_male", "sex_female",
            "apoe_noncarrier", "apoe_carrier",
        ]
        labels_display = [
            "Age <65", "Age 65-75", "Age >75",
            "Sex: Male", "Sex: Female",
            "APOE4 Non-carrier", "APOE4 Carrier",
        ]

        aucs = []
        ci_lowers = []
        ci_uppers = []
        ns = []

        for key in subgroup_keys:
            if key not in results:
                aucs.append(float("nan"))
                ci_lowers.append(float("nan"))
                ci_uppers.append(float("nan"))
                ns.append(0)
            else:
                auc = results[key]["auc"]
                ci = results[key]["ci"]
                aucs.append(auc)
                ci_lowers.append(ci[0] if not np.isnan(auc) else float("nan"))
                ci_uppers.append(ci[1] if not np.isnan(auc) else float("nan"))
                ns.append(results[key]["n"])

        aucs = np.array(aucs, dtype=float)
        ci_lowers = np.array(ci_lowers, dtype=float)
        ci_uppers = np.array(ci_uppers, dtype=float)

        # Error bars: symmetric around AUC value
        yerr_low = np.where(
            np.isnan(aucs) | np.isnan(ci_lowers),
            0,
            np.maximum(aucs - ci_lowers, 0),
        )
        yerr_high = np.where(
            np.isnan(aucs) | np.isnan(ci_uppers),
            0,
            np.maximum(ci_uppers - aucs, 0),
        )

        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(subgroup_keys))
        colors = ["#1f77b4", "#1f77b4", "#1f77b4",
                  "#ff7f0e", "#ff7f0e",
                  "#2ca02c", "#2ca02c"]

        bars = ax.bar(
            x,
            np.where(np.isnan(aucs), 0, aucs),
            yerr=[yerr_low, yerr_high],
            color=colors,
            alpha=0.8,
            edgecolor="white",
            capsize=5,
            error_kw={"elinewidth": 1.5},
        )

        # Add threshold line
        ax.axhline(y=0.80, color="red", linestyle="--", linewidth=1.5, label="ADNI target (0.80)")
        ax.axhline(y=0.78, color="orange", linestyle="--", linewidth=1.5, label="BH target (0.78)")

        # Add n labels
        for i, (bar, n) in enumerate(zip(bars, ns)):
            if not np.isnan(aucs[i]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    0.02,
                    f"n={n}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="white",
                    fontweight="bold",
                )

        ax.set_xticks(x)
        ax.set_xticklabels(labels_display, rotation=30, ha="right", fontsize=10)
        ax.set_ylabel("AUC-ROC", fontsize=12)
        ax.set_ylim([0, 1.05])

        max_gap = results.get("max_auc_gap", float("nan"))
        fairness_pass = results.get("fairness_pass", False)
        status = "PASS" if fairness_pass else "FAIL"
        gap_str = f"{max_gap:.3f}" if not np.isnan(max_gap) else "NaN"
        ax.set_title(
            f"Subgroup AUC Comparison — Max Gap: {gap_str} (Threshold <0.07: {status})",
            fontsize=13,
        )
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        log.info("Subgroup comparison plot saved", path=save_path)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid.

    Args:
        x: Input array.

    Returns:
        Sigmoid values in (0, 1).
    """
    return np.clip(1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float))), 1e-7, 1 - 1e-7)
