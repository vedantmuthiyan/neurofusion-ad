"""NeuroFusion-AD evaluation package.

Provides model evaluation, explainability, and fairness analysis tools
compliant with IEC 62304 and ISO 14971.

Modules:
    metrics:             ModelEvaluator with bootstrap CIs, format_metrics_table.
    shap_explainability: NeuralFusionSHAPExplainer (KernelExplainer, model-agnostic).
    attention_analysis:  AttentionAnalyzer for cross-modal attention weight extraction.
    subgroup_analysis:   SubgroupAnalyzer for fairness / equity analysis.
    calibration:         CalibrationEvaluator (ECE + temperature scaling).

Document traceability:
    SRS-001 § 6.1 — Evaluation Requirements
    RMF-001 § 4.2 — Performance Monitoring
"""

from src.evaluation.metrics import ModelEvaluator, format_metrics_table
from src.evaluation.calibration import CalibrationEvaluator
from src.evaluation.subgroup_analysis import SubgroupAnalyzer
from src.evaluation.attention_analysis import AttentionAnalyzer
from src.evaluation.shap_explainability import NeuralFusionSHAPExplainer

__all__ = [
    "ModelEvaluator",
    "format_metrics_table",
    "CalibrationEvaluator",
    "SubgroupAnalyzer",
    "AttentionAnalyzer",
    "NeuralFusionSHAPExplainer",
]
