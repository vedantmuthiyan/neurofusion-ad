"""Unit tests for NeuroFusion-AD evaluation infrastructure.

Tests cover:
    - ModelEvaluator: compute_all, bootstrap CI, NaN regression handling
    - CalibrationEvaluator: ECE, temperature fitting, probability output
    - SubgroupAnalyzer: analyze, edge cases (small subgroup, single class)
    - AttentionAnalyzer: modality importance scores
    - NeuralFusionSHAPExplainer: shape of shap_values
    - format_metrics_table: non-empty string output

All tests use only synthetic data — no real patient data is required.
No PHI is used or referenced.

IEC 62304 requirement traceability: SRS-001 § 6.1 (Evaluation Requirements)
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.metrics import ModelEvaluator, format_metrics_table
from src.evaluation.calibration import CalibrationEvaluator
from src.evaluation.subgroup_analysis import SubgroupAnalyzer
from src.evaluation.attention_analysis import AttentionAnalyzer


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_binary_predictions(
    n: int = 50,
    seed: int = 0,
    auc_signal: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic binary classification predictions with known signal.

    Args:
        n: Number of samples.
        seed: Random seed.
        auc_signal: Rough target AUC (higher = more separable).

    Returns:
        Tuple of (y_true, logits).
    """
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n).astype(float)
    # logits: positive class gets slightly higher values
    logits = rng.standard_normal(n) + auc_signal * (2 * y_true - 1)
    return y_true, logits


def _make_full_targets(n: int = 50, seed: int = 0) -> tuple[dict, dict]:
    """Create synthetic full predictions and targets for ModelEvaluator.

    Args:
        n: Number of samples.
        seed: Random seed.

    Returns:
        Tuple of (predictions_dict, targets_dict).
    """
    rng = np.random.default_rng(seed)
    y_true, logits = _make_binary_predictions(n=n, seed=seed)
    mmse_pred = rng.standard_normal(n).astype(np.float32)
    cox_pred = rng.standard_normal(n).astype(np.float32)
    mmse_true = rng.uniform(-3.0, 3.0, size=n).astype(np.float32)
    surv_time = rng.uniform(6.0, 60.0, size=n).astype(np.float32)
    event_ind = rng.integers(0, 2, size=n).astype(float)

    predictions = {
        "amyloid_logit": logits.reshape(-1, 1),
        "mmse_slope": mmse_pred.reshape(-1, 1),
        "cox_log_hazard": cox_pred.reshape(-1, 1),
    }
    targets = {
        "amyloid_label": y_true,
        "mmse_slope": mmse_true,
        "survival_time": surv_time,
        "event_indicator": event_ind,
    }
    return predictions, targets


def _make_metadata(n: int = 50, seed: int = 0) -> pd.DataFrame:
    """Create synthetic patient metadata for SubgroupAnalyzer.

    Args:
        n: Number of samples.
        seed: Random seed.

    Returns:
        DataFrame with AGE, SEX_CODE, APOE4_COUNT columns.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "AGE": rng.uniform(55, 85, size=n),
        "SEX_CODE": rng.integers(0, 2, size=n).astype(float),
        "APOE4_COUNT": rng.integers(0, 3, size=n).astype(float),
    })


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def evaluator() -> ModelEvaluator:
    """Return a ModelEvaluator with low bootstrap count for speed."""
    return ModelEvaluator(n_bootstrap=100, random_state=0)


@pytest.fixture
def cal_evaluator() -> CalibrationEvaluator:
    """Return a CalibrationEvaluator with default settings."""
    return CalibrationEvaluator(n_bins=10)


@pytest.fixture
def sub_analyzer() -> SubgroupAnalyzer:
    """Return a SubgroupAnalyzer with low bootstrap count for speed."""
    return SubgroupAnalyzer(n_bootstrap=100, random_state=0)


@pytest.fixture
def attn_analyzer() -> AttentionAnalyzer:
    """Return an AttentionAnalyzer."""
    return AttentionAnalyzer()


# ---------------------------------------------------------------------------
# Test 1: ModelEvaluator.compute_all returns dict with expected keys
# ---------------------------------------------------------------------------


class TestModelEvaluatorKeys:
    """Tests that ModelEvaluator.compute_all returns the correct output keys."""

    def test_compute_all_returns_expected_keys(self, evaluator: ModelEvaluator) -> None:
        """Test 1: compute_all returns dict containing all expected metric keys."""
        predictions, targets = _make_full_targets(n=60)
        result = evaluator.compute_all(predictions, targets)

        expected_keys = {
            "auc", "auc_ci", "auc_pr",
            "sensitivity", "specificity",
            "rmse", "rmse_ci", "mae", "r2",
            "c_index", "c_index_ci",
        }
        missing = expected_keys - set(result.keys())
        assert not missing, f"Missing keys in compute_all output: {missing}"


# ---------------------------------------------------------------------------
# Test 2: ModelEvaluator AUC is between 0 and 1
# ---------------------------------------------------------------------------


class TestModelEvaluatorAUC:
    """Tests that AUC values are valid probabilities."""

    def test_auc_between_0_and_1_random_data(self, evaluator: ModelEvaluator) -> None:
        """Test 2: AUC is in [0, 1] on random binary data."""
        predictions, targets = _make_full_targets(n=80, seed=1)
        result = evaluator.compute_all(predictions, targets)
        auc = result["auc"]

        assert not math.isnan(auc), "AUC should not be NaN with valid binary data"
        assert 0.0 <= auc <= 1.0, f"AUC out of [0,1]: {auc}"

    def test_auc_pr_between_0_and_1(self, evaluator: ModelEvaluator) -> None:
        """Test 2b: AUC-PR is in [0, 1] on random binary data."""
        predictions, targets = _make_full_targets(n=80, seed=2)
        result = evaluator.compute_all(predictions, targets)
        auc_pr = result["auc_pr"]

        assert not math.isnan(auc_pr), "AUC-PR should not be NaN with valid data"
        assert 0.0 <= auc_pr <= 1.0, f"AUC-PR out of [0,1]: {auc_pr}"


# ---------------------------------------------------------------------------
# Test 3: Bootstrap CI lower < upper
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    """Tests that bootstrap confidence intervals are correctly ordered."""

    def test_bootstrap_ci_lower_less_than_upper(
        self, evaluator: ModelEvaluator
    ) -> None:
        """Test 3: bootstrap_metric returns lower CI < upper CI."""
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(3)
        y_true = rng.integers(0, 2, size=100).astype(float)
        y_pred = rng.uniform(0, 1, size=100)

        _, (lower, upper) = evaluator.bootstrap_metric(
            y_true, y_pred,
            lambda yt, yp: roc_auc_score(yt, yp) if len(np.unique(yt)) > 1 else 0.5,
        )

        assert not math.isnan(lower), "CI lower should not be NaN"
        assert not math.isnan(upper), "CI upper should not be NaN"
        assert lower <= upper, (
            f"CI lower ({lower:.4f}) must be <= upper ({upper:.4f})"
        )

    def test_auc_ci_in_compute_all(self, evaluator: ModelEvaluator) -> None:
        """Test 3b: compute_all AUC CI is ordered (lower <= upper)."""
        predictions, targets = _make_full_targets(n=80, seed=4)
        result = evaluator.compute_all(predictions, targets)
        lower, upper = result["auc_ci"]

        if not math.isnan(lower) and not math.isnan(upper):
            assert lower <= upper, (
                f"AUC CI lower ({lower:.4f}) > upper ({upper:.4f})"
            )


# ---------------------------------------------------------------------------
# Test 4: ModelEvaluator handles all-NaN regression targets gracefully
# ---------------------------------------------------------------------------


class TestNaNRegressionHandling:
    """Tests NaN-tolerance in regression metric computation."""

    def test_all_nan_mmse_targets_returns_nan_rmse(
        self, evaluator: ModelEvaluator
    ) -> None:
        """Test 4: all-NaN regression targets -> NaN RMSE (no crash)."""
        rng = np.random.default_rng(5)
        n = 50
        y_true, logits = _make_binary_predictions(n=n, seed=5)
        predictions = {
            "amyloid_logit": logits,
            "mmse_slope": rng.standard_normal(n),
            "cox_log_hazard": rng.standard_normal(n),
        }
        targets = {
            "amyloid_label": y_true,
            "mmse_slope": np.full(n, float("nan")),  # all NaN
            "survival_time": rng.uniform(6.0, 60.0, size=n),
            "event_indicator": rng.integers(0, 2, size=n).astype(float),
        }

        # Should not raise
        result = evaluator.compute_all(predictions, targets)

        assert math.isnan(result["rmse"]), (
            "RMSE should be NaN when all regression targets are NaN"
        )
        assert math.isnan(result["mae"]), (
            "MAE should be NaN when all regression targets are NaN"
        )

    def test_partial_nan_mmse_computes_correctly(
        self, evaluator: ModelEvaluator
    ) -> None:
        """Test 4b: partial NaN regression targets compute on non-NaN samples."""
        rng = np.random.default_rng(6)
        n = 50
        y_true, logits = _make_binary_predictions(n=n, seed=6)
        mmse_true = rng.uniform(-3.0, 3.0, size=n)
        mmse_true[:20] = float("nan")  # 20 NaN, 30 valid

        predictions = {
            "amyloid_logit": logits,
            "mmse_slope": rng.standard_normal(n),
            "cox_log_hazard": rng.standard_normal(n),
        }
        targets = {
            "amyloid_label": y_true,
            "mmse_slope": mmse_true,
            "survival_time": rng.uniform(6.0, 60.0, size=n),
            "event_indicator": rng.integers(0, 2, size=n).astype(float),
        }

        result = evaluator.compute_all(predictions, targets)

        assert not math.isnan(result["rmse"]), (
            "RMSE should not be NaN when 30 valid regression targets exist"
        )
        assert result["rmse"] >= 0.0, "RMSE must be non-negative"


# ---------------------------------------------------------------------------
# Test 5: CalibrationEvaluator ECE between 0 and 1
# ---------------------------------------------------------------------------


class TestCalibrationECE:
    """Tests CalibrationEvaluator.compute_ece."""

    def test_ece_between_0_and_1(self, cal_evaluator: CalibrationEvaluator) -> None:
        """Test 5: ECE is in [0, 1] on random binary data."""
        rng = np.random.default_rng(7)
        probs = rng.uniform(0, 1, size=100)
        labels = rng.integers(0, 2, size=100).astype(float)
        ece = cal_evaluator.compute_ece(probs, labels)

        assert 0.0 <= ece <= 1.0, f"ECE out of [0,1]: {ece}"

    def test_ece_perfect_calibration_near_zero(
        self, cal_evaluator: CalibrationEvaluator
    ) -> None:
        """Test 5b: ECE is near 0 for a perfectly calibrated model (approx)."""
        # Create data where predicted prob ~= empirical frequency
        n = 1000
        probs = np.linspace(0.05, 0.95, n)
        rng = np.random.default_rng(8)
        labels = rng.binomial(1, probs).astype(float)
        ece = cal_evaluator.compute_ece(probs, labels)

        # Tolerance of 0.05 for finite sample noise
        assert ece < 0.05, f"ECE for calibrated model should be near 0, got {ece:.4f}"

    def test_ece_raises_on_mismatched_lengths(
        self, cal_evaluator: CalibrationEvaluator
    ) -> None:
        """Test 5c: compute_ece raises ValueError on mismatched array lengths."""
        probs = np.array([0.1, 0.5, 0.9])
        labels = np.array([0, 1])  # wrong length
        with pytest.raises(ValueError, match="length"):
            cal_evaluator.compute_ece(probs, labels)


# ---------------------------------------------------------------------------
# Test 6: CalibrationEvaluator fit_temperature returns T > 0
# ---------------------------------------------------------------------------


class TestTemperatureScaling:
    """Tests CalibrationEvaluator.fit_temperature."""

    def test_fit_temperature_returns_positive(
        self, cal_evaluator: CalibrationEvaluator
    ) -> None:
        """Test 6: fit_temperature returns a temperature > 0."""
        rng = np.random.default_rng(9)
        logits = rng.standard_normal(100)
        labels = rng.integers(0, 2, size=100).astype(float)
        temperature = cal_evaluator.fit_temperature(logits, labels)

        assert temperature > 0.0, f"Temperature must be > 0, got {temperature}"

    def test_fit_temperature_returns_float(
        self, cal_evaluator: CalibrationEvaluator
    ) -> None:
        """Test 6b: fit_temperature returns a Python float."""
        rng = np.random.default_rng(10)
        logits = rng.standard_normal(80)
        labels = rng.integers(0, 2, size=80).astype(float)
        temperature = cal_evaluator.fit_temperature(logits, labels)

        assert isinstance(temperature, float), (
            f"Expected float, got {type(temperature)}"
        )


# ---------------------------------------------------------------------------
# Test 7: apply_temperature produces valid probabilities
# ---------------------------------------------------------------------------


class TestApplyTemperature:
    """Tests CalibrationEvaluator.apply_temperature."""

    def test_apply_temperature_probabilities_in_0_1(
        self, cal_evaluator: CalibrationEvaluator
    ) -> None:
        """Test 7: apply_temperature returns probabilities in [0, 1]."""
        rng = np.random.default_rng(11)
        logits = rng.standard_normal(100) * 5  # wide range
        cal_probs = cal_evaluator.apply_temperature(logits, temperature=1.5)

        assert cal_probs.shape == (100,), f"Shape mismatch: {cal_probs.shape}"
        assert cal_probs.min() >= 0.0, f"Probabilities below 0: {cal_probs.min()}"
        assert cal_probs.max() <= 1.0, f"Probabilities above 1: {cal_probs.max()}"

    def test_apply_temperature_raises_on_invalid_T(
        self, cal_evaluator: CalibrationEvaluator
    ) -> None:
        """Test 7b: apply_temperature raises ValueError for T <= 0."""
        logits = np.array([0.5, -0.5, 1.0])
        with pytest.raises(ValueError, match="Temperature"):
            cal_evaluator.apply_temperature(logits, temperature=-1.0)

    def test_apply_temperature_T1_equals_sigmoid(
        self, cal_evaluator: CalibrationEvaluator
    ) -> None:
        """Test 7c: apply_temperature(T=1.0) equals standard sigmoid."""
        logits = np.array([2.0, -1.0, 0.0])
        result = cal_evaluator.apply_temperature(logits, temperature=1.0)
        expected = 1.0 / (1.0 + np.exp(-logits))
        np.testing.assert_allclose(result, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test 8: SubgroupAnalyzer returns dict with all expected subgroup keys
# ---------------------------------------------------------------------------


class TestSubgroupAnalyzerKeys:
    """Tests SubgroupAnalyzer.analyze output structure."""

    def test_analyze_returns_all_expected_keys(
        self, sub_analyzer: SubgroupAnalyzer
    ) -> None:
        """Test 8: analyze returns dict with all 7 subgroup keys + max_auc_gap + fairness_pass."""
        n = 100
        rng = np.random.default_rng(12)
        y_true = rng.integers(0, 2, size=n).astype(float)
        logits = rng.standard_normal(n) + (2 * y_true - 1) * 0.8

        model_outputs = {"amyloid_logit": logits}
        targets = {"amyloid_label": y_true}
        metadata = _make_metadata(n=n, seed=12)

        result = sub_analyzer.analyze(model_outputs, targets, metadata)

        expected_keys = {
            "age_lt65", "age_65_75", "age_gt75",
            "sex_male", "sex_female",
            "apoe_noncarrier", "apoe_carrier",
            "max_auc_gap", "fairness_pass",
        }
        missing = expected_keys - set(result.keys())
        assert not missing, f"Missing keys in subgroup analysis: {missing}"

    def test_analyze_fairness_pass_is_bool(
        self, sub_analyzer: SubgroupAnalyzer
    ) -> None:
        """Test 8b: fairness_pass field is a bool."""
        n = 100
        rng = np.random.default_rng(13)
        y_true = rng.integers(0, 2, size=n).astype(float)
        logits = rng.standard_normal(n)
        model_outputs = {"amyloid_logit": logits}
        targets = {"amyloid_label": y_true}
        metadata = _make_metadata(n=n, seed=13)

        result = sub_analyzer.analyze(model_outputs, targets, metadata)
        assert isinstance(result["fairness_pass"], bool), (
            f"fairness_pass must be bool, got {type(result['fairness_pass'])}"
        )


# ---------------------------------------------------------------------------
# Test 9: SubgroupAnalyzer handles n < 5 without crashing
# ---------------------------------------------------------------------------


class TestSubgroupSmallN:
    """Tests SubgroupAnalyzer with small subgroup sizes."""

    def test_small_subgroup_returns_nan_auc(
        self, sub_analyzer: SubgroupAnalyzer
    ) -> None:
        """Test 9: subgroup with n < 5 patients returns NaN AUC without crash."""
        n = 20
        rng = np.random.default_rng(14)
        y_true = rng.integers(0, 2, size=n).astype(float)
        logits = rng.standard_normal(n)

        # Create metadata where almost all patients are in one age group
        # Only 2 patients are age > 75 — too small for AUC
        metadata = pd.DataFrame({
            "AGE": np.array([60.0] * 18 + [80.0, 82.0]),  # 18 age 60, 2 age 80+
            "SEX_CODE": rng.integers(0, 2, size=n).astype(float),
            "APOE4_COUNT": rng.integers(0, 3, size=n).astype(float),
        })

        model_outputs = {"amyloid_logit": logits}
        targets = {"amyloid_label": y_true}

        # Should not raise
        result = sub_analyzer.analyze(model_outputs, targets, metadata)

        # age_gt75 has only 2 patients — must be NaN
        age_gt75 = result.get("age_gt75", {})
        assert age_gt75["n"] == 2, f"Expected n=2 for age_gt75, got {age_gt75['n']}"
        assert math.isnan(age_gt75["auc"]), (
            f"AUC for tiny subgroup should be NaN, got {age_gt75['auc']}"
        )

    def test_missing_metadata_column_raises_key_error(
        self, sub_analyzer: SubgroupAnalyzer
    ) -> None:
        """Test 9b: missing metadata column raises KeyError."""
        n = 30
        rng = np.random.default_rng(15)
        y_true = rng.integers(0, 2, size=n).astype(float)
        logits = rng.standard_normal(n)

        # Missing APOE4_COUNT column
        metadata = pd.DataFrame({
            "AGE": rng.uniform(55, 85, size=n),
            "SEX_CODE": rng.integers(0, 2, size=n).astype(float),
            # APOE4_COUNT missing intentionally
        })

        with pytest.raises(KeyError, match="APOE4_COUNT"):
            sub_analyzer.analyze(
                {"amyloid_logit": logits},
                {"amyloid_label": y_true},
                metadata,
            )


# ---------------------------------------------------------------------------
# Test 10: AttentionAnalyzer modality importance sums to ~1.0
# ---------------------------------------------------------------------------


class TestAttentionAnalyzerImportance:
    """Tests AttentionAnalyzer.get_modality_importance_scores."""

    def test_modality_importance_returns_4_values(
        self, attn_analyzer: AttentionAnalyzer
    ) -> None:
        """Test 10: get_modality_importance_scores returns 4 values."""
        # Create synthetic attention results directly (no broadcast needed)
        rng = np.random.default_rng(16)
        attn_weights = rng.dirichlet(np.ones(4), size=(20, 8, 4))
        # Tile last dim to make [N, n_heads, 4, 4]
        attn_weights_4d = np.stack([attn_weights] * 4, axis=-1)

        attention_results = {
            "attention_weights": attn_weights_4d,
            "modality_importance": np.array([0.31, 0.24, 0.22, 0.23]),
            "modality_names": ["fluid", "acoustic", "motor", "clinical"],
        }

        scores = attn_analyzer.get_modality_importance_scores(attention_results)

        assert len(scores) == 4, f"Expected 4 importance scores, got {len(scores)}"
        assert set(scores.keys()) == {"fluid", "acoustic", "motor", "clinical"}, (
            f"Unexpected keys: {set(scores.keys())}"
        )

    def test_modality_importance_sums_to_1(
        self, attn_analyzer: AttentionAnalyzer
    ) -> None:
        """Test 10b: normalized modality importance sums to ~1.0."""
        rng = np.random.default_rng(17)
        # Create importance scores that sum to 1
        raw = rng.dirichlet(np.ones(4))

        attention_results = {
            "attention_weights": np.zeros((10, 8, 4, 4)),
            "modality_importance": raw,
            "modality_names": ["fluid", "acoustic", "motor", "clinical"],
        }

        scores = attn_analyzer.get_modality_importance_scores(attention_results)
        total = sum(scores.values())

        assert abs(total - 1.0) < 1e-5, (
            f"Modality importance scores should sum to 1.0, got {total:.6f}"
        )


# ---------------------------------------------------------------------------
# Test 11: NeuralFusionSHAPExplainer returns shap_values with shape [n, 36]
# ---------------------------------------------------------------------------


class TestSHAPExplainer:
    """Tests NeuralFusionSHAPExplainer using a mocked shap.KernelExplainer."""

    def _make_mock_shap_module(self, n_explain: int = 5) -> MagicMock:
        """Create a mock shap module with a KernelExplainer that returns zeros.

        Args:
            n_explain: Number of test samples (shapes the mock return value).

        Returns:
            MagicMock configured to behave as a shap module.
        """
        mock_shap_module = MagicMock()
        mock_explainer_instance = MagicMock()
        mock_shap_values = np.zeros((n_explain, 36), dtype=np.float32)
        mock_explainer_instance.shap_values.return_value = mock_shap_values
        mock_explainer_instance.expected_value = 0.3
        mock_shap_module.KernelExplainer.return_value = mock_explainer_instance
        mock_shap_module.Explanation = MagicMock()
        return mock_shap_module

    def test_explain_returns_correct_shap_shape(self) -> None:
        """Test 11: explain returns shap_values with shape [n_samples, 36]."""
        import sys
        import importlib
        from src.models.neurofusion_model import NeuroFusionAD
        from src.data.dataset import generate_synthetic_adni
        from torch.utils.data import DataLoader

        # Build a small dataset
        dataset = generate_synthetic_adni(n_samples=30, seed=99)
        bg_loader = DataLoader(dataset, batch_size=10)
        bg_batches = [next(iter(bg_loader))]

        test_loader = DataLoader(dataset, batch_size=5)
        test_batches = [next(iter(test_loader))]

        model = NeuroFusionAD()
        model.eval()

        n_explain = 5
        mock_shap = self._make_mock_shap_module(n_explain)

        # Inject mock into sys.modules so module-level 'shap' import resolves to mock
        sys.modules["shap"] = mock_shap
        # Force reimport of the module so it picks up the mock
        if "src.evaluation.shap_explainability" in sys.modules:
            del sys.modules["src.evaluation.shap_explainability"]
        if "src.evaluation" in sys.modules:
            del sys.modules["src.evaluation"]

        from src.evaluation.shap_explainability import NeuralFusionSHAPExplainer

        try:
            explainer = NeuralFusionSHAPExplainer(
                model=model,
                background_data=bg_batches,
                device="cpu",
            )
            results = explainer.explain(test_batches, n_samples=n_explain)
        finally:
            # Restore: remove mock from sys.modules
            del sys.modules["shap"]
            if "src.evaluation.shap_explainability" in sys.modules:
                del sys.modules["src.evaluation.shap_explainability"]

        assert "shap_values" in results, "explain() must return 'shap_values' key"
        shap_vals = results["shap_values"]
        assert shap_vals.shape == (n_explain, 36), (
            f"Expected shap_values shape ({n_explain}, 36), got {shap_vals.shape}"
        )

    def test_explain_returns_feature_names(self) -> None:
        """Test 11b: explain returns list of 36 feature names."""
        import sys
        from src.models.neurofusion_model import NeuroFusionAD
        from src.data.dataset import generate_synthetic_adni
        from torch.utils.data import DataLoader

        dataset = generate_synthetic_adni(n_samples=20, seed=88)
        bg_loader = DataLoader(dataset, batch_size=10)
        bg_batches = [next(iter(bg_loader))]
        test_batches = [next(iter(DataLoader(dataset, batch_size=5)))]

        model = NeuroFusionAD()
        model.eval()

        mock_shap = self._make_mock_shap_module(n_explain=5)

        sys.modules["shap"] = mock_shap
        if "src.evaluation.shap_explainability" in sys.modules:
            del sys.modules["src.evaluation.shap_explainability"]

        from src.evaluation.shap_explainability import NeuralFusionSHAPExplainer

        try:
            explainer = NeuralFusionSHAPExplainer(
                model=model,
                background_data=bg_batches,
                device="cpu",
            )
            results = explainer.explain(test_batches, n_samples=5)
        finally:
            del sys.modules["shap"]
            if "src.evaluation.shap_explainability" in sys.modules:
                del sys.modules["src.evaluation.shap_explainability"]

        feature_names = results["feature_names"]
        assert len(feature_names) == 36, (
            f"Expected 36 feature names, got {len(feature_names)}"
        )
        assert all(isinstance(fn, str) for fn in feature_names), (
            "All feature names must be strings"
        )


# ---------------------------------------------------------------------------
# Test 12: format_metrics_table returns non-empty string
# ---------------------------------------------------------------------------


class TestFormatMetricsTable:
    """Tests format_metrics_table output."""

    def test_format_metrics_table_nonempty(self) -> None:
        """Test 12: format_metrics_table returns a non-empty string."""
        metrics = {
            "auc": 0.87,
            "auc_ci": (0.82, 0.91),
            "rmse": 2.3,
            "c_index": 0.74,
        }
        result = format_metrics_table(metrics)

        assert isinstance(result, str), f"Expected str, got {type(result)}"
        assert len(result) > 0, "format_metrics_table returned empty string"

    def test_format_metrics_table_contains_header(self) -> None:
        """Test 12b: format_metrics_table includes markdown table header."""
        metrics = {"auc": 0.85}
        result = format_metrics_table(metrics)

        assert "Metric" in result, "Table should contain 'Metric' header"
        assert "Value" in result, "Table should contain 'Value' header"
        assert "|" in result, "Table should use | as column separator"

    def test_format_metrics_table_empty_dict(self) -> None:
        """Test 12c: format_metrics_table handles empty dict without crash."""
        result = format_metrics_table({})

        assert isinstance(result, str)
        assert len(result) > 0, "Should return placeholder row for empty dict"

    def test_format_metrics_table_ci_formatted_as_list(self) -> None:
        """Test 12d: CI values are formatted as [lower, upper] in table."""
        metrics = {"auc_ci": (0.81, 0.89)}
        result = format_metrics_table(metrics)

        assert "0.8100" in result or "0.81" in result, (
            f"Lower CI not found in table: {result}"
        )
        assert "0.8900" in result or "0.89" in result, (
            f"Upper CI not found in table: {result}"
        )

    def test_format_metrics_table_nan_handled(self) -> None:
        """Test 12e: NaN values are represented as 'NaN' string (no crash)."""
        metrics = {"rmse": float("nan"), "auc": 0.85}
        result = format_metrics_table(metrics)

        assert "NaN" in result, f"NaN should appear as 'NaN' in table: {result}"
