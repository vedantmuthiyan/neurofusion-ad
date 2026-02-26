"""Input validation for NeuroFusion-AD multimodal features.

Enforces hard clinical range constraints defined in the NeuroFusion-AD specification.
All validation errors are logged without feature values to preserve PHI compliance.

IEC 62304 Requirement Traceability: SRS-001 § 4.2 (Input Validation)
"""

import structlog
import torch

logger = structlog.get_logger(__name__)


class InputValidator:
    """Validates multimodal input features against clinical range constraints.

    Enforces the hard-coded validated ranges from the NeuroFusion-AD specification.
    Any feature value outside these ranges indicates either a data quality issue
    or a patient outside the intended use population (ages 50–90 with MCI).

    Raises:
        ValueError: When any feature value falls outside its validated clinical range.
    """

    # Validated ranges from CLAUDE.md (hard constraints — reject outside these)
    FLUID_RANGES = {
        "ptau217": (0.1, 100.0),         # pg/mL
        "abeta42_40_ratio": (0.01, 0.30),
        "nfl": (5.0, 200.0),             # pg/mL
    }
    ACOUSTIC_RANGES = {
        "jitter": (0.0001, 0.05),
        "shimmer": (0.001, 0.3),
    }
    CLINICAL_RANGES = {
        "mmse": (0.0, 30.0),
    }

    # Feature index maps for human-readable error messages
    _FLUID_IDX = {
        "ptau217": 0,
        "abeta42_40_ratio": 1,
        "nfl": 2,
    }
    _ACOUSTIC_IDX = {
        "jitter": 0,
        "shimmer": 1,
    }
    _CLINICAL_IDX = {
        "mmse": 3,
    }

    def validate_fluid_biomarkers(self, features: torch.Tensor) -> None:
        """Validate fluid biomarker tensor against clinical range constraints.

        Validated features (by index):
            0: ptau217 — range [0.1, 100.0] pg/mL
            1: abeta42_40_ratio — range [0.01, 0.30]
            2: nfl — range [5.0, 200.0] pg/mL
            3: gfap — no hard constraint
            4: total_tau — no hard constraint
            5: abeta42 — no hard constraint

        Args:
            features: Fluid biomarker tensor of shape [6] or [batch, 6].

        Raises:
            ValueError: If any validated feature is outside its clinical range.
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)

        for name, (lo, hi) in self.FLUID_RANGES.items():
            idx = self._FLUID_IDX[name]
            col = features[:, idx]
            if torch.any(col < lo) or torch.any(col > hi):
                logger.warning(
                    "fluid_biomarker_out_of_range",
                    feature=name,
                    valid_min=lo,
                    valid_max=hi,
                    n_violations=int(((col < lo) | (col > hi)).sum().item()),
                )
                raise ValueError(
                    f"Fluid biomarker '{name}' (index {idx}) contains values outside "
                    f"validated clinical range [{lo}, {hi}]. "
                    "Ensure patient is within the intended use population."
                )

    def validate_acoustic_features(self, features: torch.Tensor) -> None:
        """Validate acoustic feature tensor against clinical range constraints.

        Validated features (by index):
            0: jitter — range [0.0001, 0.05]
            1: shimmer — range [0.001, 0.3]
            2: hnr — no hard constraint
            3: f0_mean — no hard constraint
            4: f0_std — no hard constraint
            5–11: mfcc_1..7 — no hard constraints

        Args:
            features: Acoustic feature tensor of shape [12] or [batch, 12].

        Raises:
            ValueError: If any validated feature is outside its clinical range.
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)

        for name, (lo, hi) in self.ACOUSTIC_RANGES.items():
            idx = self._ACOUSTIC_IDX[name]
            col = features[:, idx]
            if torch.any(col < lo) or torch.any(col > hi):
                logger.warning(
                    "acoustic_feature_out_of_range",
                    feature=name,
                    valid_min=lo,
                    valid_max=hi,
                    n_violations=int(((col < lo) | (col > hi)).sum().item()),
                )
                raise ValueError(
                    f"Acoustic feature '{name}' (index {idx}) contains values outside "
                    f"validated clinical range [{lo}, {hi}]. "
                    "Check acoustic preprocessing pipeline."
                )

    def validate_clinical_features(self, features: torch.Tensor) -> None:
        """Validate clinical/demographic feature tensor against range constraints.

        Validated features (by index):
            0: age — no hard constraint (population filter applied upstream)
            1: education — no hard constraint
            2: sex — no hard constraint
            3: mmse — range [0.0, 30.0]
            4: cdr — no hard constraint
            5: gds — no hard constraint
            6: bmi — no hard constraint
            7: systolic_bp — no hard constraint
            8: apoe4 — no hard constraint
            9: comorbidities — no hard constraint

        Args:
            features: Clinical feature tensor of shape [10] or [batch, 10].

        Raises:
            ValueError: If any validated feature is outside its clinical range.
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)

        for name, (lo, hi) in self.CLINICAL_RANGES.items():
            idx = self._CLINICAL_IDX[name]
            col = features[:, idx]
            if torch.any(col < lo) or torch.any(col > hi):
                logger.warning(
                    "clinical_feature_out_of_range",
                    feature=name,
                    valid_min=lo,
                    valid_max=hi,
                    n_violations=int(((col < lo) | (col > hi)).sum().item()),
                )
                raise ValueError(
                    f"Clinical feature '{name}' (index {idx}) contains values outside "
                    f"validated clinical range [{lo}, {hi}]. "
                    "MMSE must be a standard score between 0 and 30."
                )

    def validate_all(
        self,
        fluid: torch.Tensor,
        acoustic: torch.Tensor,
        motor: torch.Tensor,
        clinical: torch.Tensor,
    ) -> None:
        """Run all modality validators sequentially.

        Validates fluid biomarkers, acoustic features, and clinical features.
        Motor features currently have no hard-coded range constraints in the
        specification and are accepted as-is.

        Args:
            fluid: Fluid biomarker tensor of shape [6] or [batch, 6].
            acoustic: Acoustic feature tensor of shape [12] or [batch, 12].
            motor: Motor feature tensor of shape [8] or [batch, 8].
            clinical: Clinical feature tensor of shape [10] or [batch, 10].

        Raises:
            ValueError: With a descriptive message on the first validation failure.
        """
        logger.debug("running_full_input_validation")
        self.validate_fluid_biomarkers(fluid)
        self.validate_acoustic_features(acoustic)
        self.validate_clinical_features(clinical)
        logger.debug("input_validation_passed")
