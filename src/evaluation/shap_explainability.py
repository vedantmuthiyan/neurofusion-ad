"""SHAP explainability for NeuroFusion-AD using model-agnostic KernelExplainer.

Uses shap.KernelExplainer (NOT DeepExplainer) for regulatory compliance.
KernelExplainer is model-agnostic and provides SHAP values that are
independent of the model's internal implementation, which is required for
SaMD submissions per FDA AI/ML guidance.

Feature layout (36 total input features):
    fluid[0-5]:    ptau217, abeta42_40_ratio, nfl_plasma, gfap_plasma,
                   ptau181_csf, abeta42_csf
    acoustic[0-11]: jitter, shimmer, hnr, f0_mean, f0_std, mfcc1-7
    motor[0-7]:    tremor_freq, tremor_amp, bradykinesia, spiral_rmse,
                   tapping_cv, tapping_asym, grip_mean, grip_cv
    clinical[0-9]: age, sex, education, mmse_baseline, apoe4, tau_csf,
                   abeta42_plasma, abeta40_plasma, ptau_tau_ratio,
                   abeta_ptau_ratio

IEC 62304 compliance:
    - No PHI logged; patient IDs must be hashed before passing.
    - KernelExplainer only (DeepExplainer is NOT approved for regulatory use
      in this project — see SAD-001 § 5.5).

Document traceability:
    SRS-001 § 6.4 — Explainability Requirements
    SAD-001 § 5.5 — SHAP Explainability Layer
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import structlog

log = structlog.get_logger(__name__)

try:
    import shap as shap
except ImportError:
    shap = None  # type: ignore[assignment]

# Feature dimension splits
FLUID_DIM = 6
ACOUSTIC_DIM = 12
MOTOR_DIM = 8
CLINICAL_DIM = 10
TOTAL_FEATURES = FLUID_DIM + ACOUSTIC_DIM + MOTOR_DIM + CLINICAL_DIM  # 36

FLUID_FEATURE_NAMES = [
    "ptau217", "abeta42_40_ratio", "nfl_plasma", "gfap_plasma",
    "ptau181_csf", "abeta42_csf",
]
ACOUSTIC_FEATURE_NAMES = [
    "jitter", "shimmer", "hnr", "f0_mean", "f0_std",
    "mfcc1", "mfcc2", "mfcc3", "mfcc4", "mfcc5", "mfcc6", "mfcc7",
]
MOTOR_FEATURE_NAMES = [
    "tremor_freq", "tremor_amp", "bradykinesia", "spiral_rmse",
    "tapping_cv", "tapping_asym", "grip_mean", "grip_cv",
]
CLINICAL_FEATURE_NAMES = [
    "age", "sex", "education", "mmse_baseline", "apoe4", "tau_csf",
    "abeta42_plasma", "abeta40_plasma", "ptau_tau_ratio", "abeta_ptau_ratio",
]

ALL_FEATURE_NAMES = (
    [f"fluid_{n}" for n in FLUID_FEATURE_NAMES]
    + [f"acoustic_{n}" for n in ACOUSTIC_FEATURE_NAMES]
    + [f"motor_{n}" for n in MOTOR_FEATURE_NAMES]
    + [f"clinical_{n}" for n in CLINICAL_FEATURE_NAMES]
)


class NeuralFusionSHAPExplainer:
    """SHAP explanations using KernelExplainer (model-agnostic, regulatory-compliant).

    Computes SHAP values for the amyloid classification head.
    Uses a background dataset of 50 samples to estimate the baseline.
    Explains predictions on a subset of test samples.

    The model expects a batch dict, but KernelExplainer expects a numpy
    array. An internal wrapper function handles the conversion:
        - Input: numpy array [n, 36]
        - Split into fluid[6], acoustic[12], motor[8], clinical[10]
        - Create batch dict, run model forward
        - Return amyloid probability as numpy array [n]

    Attributes:
        model: NeuroFusionAD instance (set to eval mode).
        device: Torch device string ('cpu' or 'cuda').
        feature_names: List of 36 feature name strings.

    Example:
        >>> explainer = NeuralFusionSHAPExplainer(model, background_data)
        >>> results = explainer.explain(test_samples, n_samples=20)
        >>> print(results['shap_values'].shape)  # (20, 36)
    """

    def __init__(
        self,
        model: nn.Module,
        background_data: list,
        device: str = "cpu",
    ) -> None:
        """Initialise NeuralFusionSHAPExplainer.

        Args:
            model: NeuroFusionAD instance. Will be set to eval mode.
            background_data: List of batch dicts from DataLoader. Used to
                construct the SHAP background dataset (max 50 samples).
            device: Device string 'cpu' or 'cuda' (default 'cpu').
                KernelExplainer works on CPU only.
        """
        # Use module-level shap (imported at top of file, or None if not installed)
        if shap is None:
            raise ImportError(
                "shap library is required. Install with: pip install shap"
            )

        self.model = model
        self.model.eval()
        self.device = device
        self.feature_names = ALL_FEATURE_NAMES

        # Flatten background_data into numpy array [n, 36]
        background_array = self._batches_to_numpy(background_data)
        # Limit to 50 samples for speed
        if background_array.shape[0] > 50:
            rng = np.random.default_rng(42)
            idx = rng.choice(background_array.shape[0], size=50, replace=False)
            background_array = background_array[idx]

        self._background = background_array

        # Build KernelExplainer with the model wrapper
        self.explainer = shap.KernelExplainer(
            self._model_wrapper, background_array
        )

        log.info(
            "NeuralFusionSHAPExplainer initialised",
            n_background=background_array.shape[0],
            total_features=TOTAL_FEATURES,
            device=device,
        )

    def _model_wrapper(self, X: np.ndarray) -> np.ndarray:
        """Wrapper converting numpy array to model input dict and back.

        Args:
            X: numpy array of shape [n, 36] with concatenated features.

        Returns:
            Amyloid probabilities as numpy array of shape [n].
        """
        n = X.shape[0]
        fluid = torch.tensor(
            X[:, :FLUID_DIM], dtype=torch.float32
        )
        acoustic = torch.tensor(
            X[:, FLUID_DIM:FLUID_DIM + ACOUSTIC_DIM], dtype=torch.float32
        )
        motor = torch.tensor(
            X[:, FLUID_DIM + ACOUSTIC_DIM:FLUID_DIM + ACOUSTIC_DIM + MOTOR_DIM],
            dtype=torch.float32,
        )
        clinical = torch.tensor(
            X[:, FLUID_DIM + ACOUSTIC_DIM + MOTOR_DIM:], dtype=torch.float32
        )

        batch = {
            "fluid": fluid,
            "acoustic": acoustic,
            "motor": motor,
            "clinical": clinical,
        }

        with torch.no_grad():
            outputs = self.model(batch)
            logits = outputs["amyloid_logit"].cpu().numpy().ravel()

        probs = _sigmoid(logits)
        return probs

    def explain(
        self,
        test_samples: list,
        n_samples: int = 20,
    ) -> dict:
        """Compute SHAP values for test samples.

        Uses KernelExplainer with nsamples=200 for speed. For regulatory
        submission, nsamples=2000 should be used.

        Args:
            test_samples: List of batch dicts from DataLoader.
            n_samples: Maximum number of test samples to explain (default 20).

        Returns:
            Dict with:
                {
                    'shap_values': ndarray[n, 36],
                    'feature_names': list[str] (36 feature names),
                    'base_value': float,
                    'predictions': ndarray[n] (amyloid probabilities),
                }
        """
        # Flatten test_samples to numpy
        test_array = self._batches_to_numpy(test_samples)
        if test_array.shape[0] > n_samples:
            test_array = test_array[:n_samples]

        n = test_array.shape[0]
        log.info("NeuralFusionSHAPExplainer.explain starting", n_test=n)

        # Compute SHAP values
        shap_values = self.explainer.shap_values(
            test_array, nsamples=200, silent=True
        )

        if isinstance(shap_values, list):
            # Multi-output: take first element (positive class)
            shap_values = shap_values[0]

        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)

        predictions = self._model_wrapper(test_array)
        base_value = float(self.explainer.expected_value)
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(np.asarray(base_value).ravel()[0])

        log.info(
            "NeuralFusionSHAPExplainer.explain complete",
            shap_shape=list(shap_values.shape),
            base_value=base_value,
        )

        return {
            "shap_values": shap_values,
            "feature_names": self.feature_names,
            "base_value": base_value,
            "predictions": predictions,
        }

    def plot_summary(
        self,
        shap_results: dict,
        save_path: str,
    ) -> None:
        """Save SHAP summary beeswarm plot.

        Args:
            shap_results: Dict returned by explain().
            save_path: Absolute path to save the PNG figure.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        shap_values = shap_results["shap_values"]
        feature_names = shap_results["feature_names"]

        # Create Explanation object for modern shap API
        base_value = shap_results["base_value"]
        explanation = shap.Explanation(
            values=shap_values,
            base_values=np.full(shap_values.shape[0], base_value),
            feature_names=feature_names,
        )

        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(explanation, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close("all")

        log.info("SHAP summary plot saved", path=save_path)

    def plot_waterfall(
        self,
        shap_results: dict,
        sample_idx: int,
        save_path: str,
    ) -> None:
        """Save SHAP waterfall plot for one sample.

        Args:
            shap_results: Dict returned by explain().
            sample_idx: Index of the sample within shap_results.
            save_path: Absolute path to save the PNG figure.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        shap_values = shap_results["shap_values"]
        feature_names = shap_results["feature_names"]
        base_value = shap_results["base_value"]
        prediction = float(shap_results["predictions"][sample_idx])

        explanation = shap.Explanation(
            values=shap_values[sample_idx],
            base_values=base_value,
            data=None,
            feature_names=feature_names,
        )

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False)
        plt.title(
            f"SHAP Waterfall — Sample {sample_idx} "
            f"(Amyloid Prob: {prediction:.3f})",
            fontsize=12,
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close("all")

        log.info(
            "SHAP waterfall plot saved",
            path=save_path,
            sample_idx=sample_idx,
        )

    def _batches_to_numpy(self, batches: list) -> np.ndarray:
        """Flatten list of batch dicts into a numpy array of shape [N, 36].

        Args:
            batches: List of batch dicts with keys 'fluid', 'acoustic',
                'motor', 'clinical'. Values can be tensors or numpy arrays.
                Each dict may contain a batch of samples.

        Returns:
            numpy array of shape [N, 36] with concatenated features.
        """
        rows = []
        for batch in batches:
            fluid = _to_numpy(batch["fluid"])
            acoustic = _to_numpy(batch["acoustic"])
            motor = _to_numpy(batch["motor"])
            clinical = _to_numpy(batch["clinical"])

            # Handle both batched [B, D] and single-sample [D] inputs
            if fluid.ndim == 1:
                fluid = fluid.reshape(1, -1)
                acoustic = acoustic.reshape(1, -1)
                motor = motor.reshape(1, -1)
                clinical = clinical.reshape(1, -1)

            concat = np.concatenate([fluid, acoustic, motor, clinical], axis=1)
            rows.append(concat)

        if not rows:
            return np.zeros((0, TOTAL_FEATURES), dtype=np.float32)

        return np.concatenate(rows, axis=0).astype(np.float32)


def _to_numpy(x) -> np.ndarray:
    """Convert tensor or array-like to numpy array.

    Args:
        x: Tensor, list, or numpy array.

    Returns:
        numpy float32 array.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid.

    Args:
        x: Input array.

    Returns:
        Probabilities in (0, 1).
    """
    return np.clip(1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float))), 1e-7, 1 - 1e-7)
