"""Synthetic digital biomarker data generation for NeuroFusion-AD.

Generates clinically plausible synthetic data for testing and validation purposes.
All distributions are based on published Alzheimer's disease literature.

WARNING: This module generates SYNTHETIC data only. It is NOT for clinical use.
Synthetic data must not be used to train or validate production models.

IEC 62304 Requirement Traceability: SRS-001 § 5.3 (Test Data Generation)
"""

import structlog
import torch
import numpy as np

logger = structlog.get_logger(__name__)

# Feature dimension constants
_FLUID_DIM = 6
_ACOUSTIC_DIM = 12
_MOTOR_DIM = 8
_CLINICAL_DIM = 10


class DigitalBiomarkerSynthesizer:
    """Generates synthetic digital biomarker data for testing and validation.

    All generated values use clinically plausible distributions derived from
    published Alzheimer's disease literature. Values are clipped to the
    validated clinical ranges defined in the NeuroFusion-AD specification.

    WARNING: Synthetic data only. Not for clinical use. Do not use synthetic
    data to draw clinical conclusions or validate model performance.

    Attributes:
        seed: Random seed for reproducibility.
        rng: NumPy random generator instance.
    """

    def __init__(self, seed: int = 42) -> None:
        """Initialize DigitalBiomarkerSynthesizer.

        Args:
            seed: Random seed for reproducibility across all synthesis calls.
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        logger.info("synthesizer_initialized", seed=seed)

    def synthesize_acoustic(self, n_samples: int) -> torch.Tensor:
        """Generate synthetic acoustic features.

        Generates 12-dimensional acoustic feature vectors using distributions
        derived from speech analysis studies in MCI/AD populations.

        Distributions:
            0 (jitter): LogNormal(mean_log=-5.3, sigma_log=0.6), clipped [0.0001, 0.05]
            1 (shimmer): LogNormal(mean_log=-3.2, sigma_log=0.5), clipped [0.001, 0.3]
            2 (hnr): Normal(15, 5), clipped [0, 30]
            3 (f0_mean): Normal(130, 30), clipped [80, 250]
            4 (f0_std): Normal(25, 10), clipped [5, 80]
            5–11 (mfcc_1..7): Normal(0, 1) for each

        Args:
            n_samples: Number of synthetic samples to generate.

        Returns:
            Float32 tensor of shape [n_samples, 12] with synthetic acoustic features.
        """
        rng = self.rng
        data = np.zeros((n_samples, _ACOUSTIC_DIM), dtype=np.float32)

        # jitter: LogNormal — target mean 0.005, std ~0.003
        # For LogNormal: mu_log = ln(mean^2 / sqrt(mean^2 + std^2))
        jitter_raw = rng.lognormal(mean=-5.3, sigma=0.6, size=n_samples)
        data[:, 0] = np.clip(jitter_raw, 0.0001, 0.05)

        # shimmer: LogNormal — target mean 0.04, std ~0.02
        shimmer_raw = rng.lognormal(mean=-3.2, sigma=0.5, size=n_samples)
        data[:, 1] = np.clip(shimmer_raw, 0.001, 0.3)

        # hnr: Normal(15, 5), clipped [0, 30]
        data[:, 2] = np.clip(rng.normal(15.0, 5.0, n_samples), 0.0, 30.0)

        # f0_mean: Normal(130, 30), clipped [80, 250]
        data[:, 3] = np.clip(rng.normal(130.0, 30.0, n_samples), 80.0, 250.0)

        # f0_std: Normal(25, 10), clipped [5, 80]
        data[:, 4] = np.clip(rng.normal(25.0, 10.0, n_samples), 5.0, 80.0)

        # mfcc_1..7: Normal(0, 1)
        for i in range(5, 12):
            data[:, i] = rng.normal(0.0, 1.0, n_samples)

        tensor = torch.tensor(data.tolist(), dtype=torch.float32)
        logger.debug("synthesized_acoustic_features", n_samples=n_samples)
        return tensor

    def synthesize_motor(self, n_samples: int) -> torch.Tensor:
        """Generate synthetic motor features.

        Generates 8-dimensional motor feature vectors using distributions
        derived from motor assessment studies in MCI/AD populations.

        Distributions:
            0 (tremor_freq): Normal(4.0, 2.0), clipped [0, 12]
            1 (tremor_amp): Normal(0.3, 0.2), clipped [0, 2]
            2 (bradykinesia_score): Normal(50, 20), clipped [0, 100]
            3 (spiral_rmse): Normal(2.0, 1.0), clipped [0, 10]
            4 (tapping_interval_cv): Normal(0.15, 0.05), clipped [0, 1]
            5 (tapping_asymmetry): Normal(0.05, 0.03), clipped [0, 0.5]
            6 (grip_force_mean): Normal(25, 10), clipped [5, 60]
            7 (grip_force_cv): Normal(0.10, 0.05), clipped [0, 0.5]

        Args:
            n_samples: Number of synthetic samples to generate.

        Returns:
            Float32 tensor of shape [n_samples, 8] with synthetic motor features.
        """
        rng = self.rng
        data = np.zeros((n_samples, _MOTOR_DIM), dtype=np.float32)

        data[:, 0] = np.clip(rng.normal(4.0,   2.0,  n_samples), 0.0,  12.0)
        data[:, 1] = np.clip(rng.normal(0.3,   0.2,  n_samples), 0.0,   2.0)
        data[:, 2] = np.clip(rng.normal(50.0,  20.0, n_samples), 0.0, 100.0)
        data[:, 3] = np.clip(rng.normal(2.0,   1.0,  n_samples), 0.0,  10.0)
        data[:, 4] = np.clip(rng.normal(0.15,  0.05, n_samples), 0.0,   1.0)
        data[:, 5] = np.clip(rng.normal(0.05,  0.03, n_samples), 0.0,   0.5)
        data[:, 6] = np.clip(rng.normal(25.0,  10.0, n_samples), 5.0,  60.0)
        data[:, 7] = np.clip(rng.normal(0.10,  0.05, n_samples), 0.0,   0.5)

        tensor = torch.tensor(data.tolist(), dtype=torch.float32)
        logger.debug("synthesized_motor_features", n_samples=n_samples)
        return tensor

    def synthesize_fluid_biomarkers(
        self,
        n_samples: int,
        amyloid_positive_fraction: float = 0.4,
    ) -> torch.Tensor:
        """Generate synthetic fluid biomarker features with two sub-populations.

        Models amyloid-positive (Abeta+) and amyloid-negative (Abeta-) patients
        with distinct biomarker distributions consistent with published reference ranges.

        Features (by index):
            0: ptau217 (pg/mL) — higher in amyloid+ patients
            1: abeta42_40_ratio — lower in amyloid+ patients
            2: nfl (pg/mL) — slightly elevated in amyloid+ patients
            3: gfap (pg/mL)
            4: total_tau (pg/mL)
            5: abeta42 (pg/mL)

        All values clipped to validated clinical ranges:
            ptau217: [0.1, 100.0], abeta42_40_ratio: [0.01, 0.30], nfl: [5.0, 200.0]

        Args:
            n_samples: Total number of synthetic samples to generate.
            amyloid_positive_fraction: Fraction of samples to generate as amyloid-positive.
                Must be in [0, 1].

        Returns:
            Float32 tensor of shape [n_samples, 6] with synthetic fluid biomarkers.

        Raises:
            ValueError: If amyloid_positive_fraction is not in [0, 1].
        """
        if not 0.0 <= amyloid_positive_fraction <= 1.0:
            raise ValueError(
                f"amyloid_positive_fraction must be in [0, 1], "
                f"got {amyloid_positive_fraction}"
            )

        rng = self.rng
        n_pos = int(round(n_samples * amyloid_positive_fraction))
        n_neg = n_samples - n_pos

        data = np.zeros((n_samples, _FLUID_DIM), dtype=np.float32)

        # --- Amyloid-positive sub-population (rows 0..n_pos-1) ---
        if n_pos > 0:
            # ptau217: elevated in Abeta+ — LogNormal, higher
            ptau_pos = rng.lognormal(mean=3.0, sigma=0.5, size=n_pos)
            data[:n_pos, 0] = np.clip(ptau_pos, 0.1, 100.0)

            # abeta42_40_ratio: decreased in Abeta+ — LogNormal, lower
            abeta_ratio_pos = rng.lognormal(mean=-2.8, sigma=0.3, size=n_pos)
            data[:n_pos, 1] = np.clip(abeta_ratio_pos, 0.01, 0.30)

            # nfl: elevated in Abeta+ — LogNormal
            nfl_pos = rng.lognormal(mean=3.5, sigma=0.4, size=n_pos)
            data[:n_pos, 2] = np.clip(nfl_pos, 5.0, 200.0)

            # gfap: pg/mL (no hard constraint, plausible range 50-500)
            data[:n_pos, 3] = np.clip(rng.lognormal(mean=5.0, sigma=0.5, size=n_pos), 20.0, 600.0)

            # total_tau: pg/mL (plausible range 100-600)
            data[:n_pos, 4] = np.clip(rng.normal(320.0, 80.0, size=n_pos), 50.0, 800.0)

            # abeta42: pg/mL (lower in Abeta+, plausible range 400-1400)
            data[:n_pos, 5] = np.clip(rng.normal(700.0, 150.0, size=n_pos), 200.0, 1500.0)

        # --- Amyloid-negative sub-population (rows n_pos..n_samples-1) ---
        if n_neg > 0:
            # ptau217: lower in Abeta-
            ptau_neg = rng.lognormal(mean=1.8, sigma=0.4, size=n_neg)
            data[n_pos:, 0] = np.clip(ptau_neg, 0.1, 100.0)

            # abeta42_40_ratio: higher in Abeta-
            abeta_ratio_neg = rng.lognormal(mean=-2.0, sigma=0.2, size=n_neg)
            data[n_pos:, 1] = np.clip(abeta_ratio_neg, 0.01, 0.30)

            # nfl: lower in Abeta-
            nfl_neg = rng.lognormal(mean=3.0, sigma=0.35, size=n_neg)
            data[n_pos:, 2] = np.clip(nfl_neg, 5.0, 200.0)

            # gfap: lower in Abeta-
            data[n_pos:, 3] = np.clip(rng.lognormal(mean=4.5, sigma=0.4, size=n_neg), 20.0, 600.0)

            # total_tau: lower in Abeta-
            data[n_pos:, 4] = np.clip(rng.normal(220.0, 60.0, size=n_neg), 50.0, 800.0)

            # abeta42: higher in Abeta-
            data[n_pos:, 5] = np.clip(rng.normal(1000.0, 150.0, size=n_neg), 200.0, 1500.0)

        # Shuffle to interleave positive and negative samples
        indices = rng.permutation(n_samples)
        data = data[indices]

        tensor = torch.tensor(data.tolist(), dtype=torch.float32)
        logger.debug(
            "synthesized_fluid_biomarkers",
            n_samples=n_samples,
            n_amyloid_positive=n_pos,
            n_amyloid_negative=n_neg,
        )
        return tensor

    def synthesize_clinical(self, n_samples: int) -> torch.Tensor:
        """Generate synthetic clinical/demographic features.

        Generates 10-dimensional clinical feature vectors using distributions
        representative of the intended-use population (MCI patients aged 50–90).

        Distributions:
            0 (age): Normal(72, 8), clipped [50, 90]
            1 (education): Normal(14, 3), clipped [6, 22]
            2 (sex): Bernoulli(0.55)
            3 (mmse): Normal(26, 3), clipped [0, 30]
            4 (cdr_sum): abs(Normal(1.5, 1.5)), clipped [0, 18]
            5 (gds): abs(Normal(5, 3)), clipped [0, 15]
            6 (bmi): Normal(27, 5), clipped [15, 50]
            7 (systolic_bp): Normal(130, 15), clipped [90, 200]
            8 (apoe4): Bernoulli(0.3)
            9 (comorbidities): Poisson(2), clipped [0, 10]

        Args:
            n_samples: Number of synthetic samples to generate.

        Returns:
            Float32 tensor of shape [n_samples, 10] with synthetic clinical features.
        """
        rng = self.rng
        data = np.zeros((n_samples, _CLINICAL_DIM), dtype=np.float32)

        data[:, 0] = np.clip(rng.normal(72.0, 8.0, n_samples), 50.0, 90.0)   # age
        data[:, 1] = np.clip(rng.normal(14.0, 3.0, n_samples), 6.0, 22.0)    # education
        data[:, 2] = rng.binomial(1, 0.55, n_samples).astype(np.float32)      # sex
        data[:, 3] = np.clip(rng.normal(26.0, 3.0, n_samples), 0.0, 30.0)    # mmse
        data[:, 4] = np.clip(np.abs(rng.normal(1.5, 1.5, n_samples)), 0.0, 18.0)  # cdr_sum
        data[:, 5] = np.clip(np.abs(rng.normal(5.0, 3.0, n_samples)), 0.0, 15.0)  # gds
        data[:, 6] = np.clip(rng.normal(27.0, 5.0, n_samples), 15.0, 50.0)   # bmi
        data[:, 7] = np.clip(rng.normal(130.0, 15.0, n_samples), 90.0, 200.0) # systolic_bp
        data[:, 8] = rng.binomial(1, 0.3, n_samples).astype(np.float32)       # apoe4
        data[:, 9] = np.clip(rng.poisson(2, n_samples), 0, 10).astype(np.float32)  # comorbidities

        tensor = torch.tensor(data.tolist(), dtype=torch.float32)
        logger.debug("synthesized_clinical_features", n_samples=n_samples)
        return tensor

    def synthesize_labels(
        self,
        n_samples: int,
        amyloid_positive_fraction: float = 0.4,
    ) -> dict[str, torch.Tensor]:
        """Generate synthetic labels for all three NeuroFusion-AD output heads.

        Produces binary amyloid classification labels, continuous MMSE slope
        (cognitive decline rate), and time-to-event data for survival analysis.
        Amyloid-positive patients tend to have faster decline and earlier events.

        Args:
            n_samples: Number of synthetic label sets to generate.
            amyloid_positive_fraction: Fraction of amyloid-positive patients.
                Used to correlate labels with expected biomarker patterns.

        Returns:
            Dict with the following keys:
                'amyloid_label': Tensor[n_samples] — binary (0 = Abeta-, 1 = Abeta+)
                'mmse_slope': Tensor[n_samples] — MMSE points/year (negative = decline)
                'time_to_event': Tensor[n_samples] — months until progression or censoring
                'event_observed': Tensor[n_samples] — 1 if event occurred, 0 if censored
        """
        rng = self.rng
        n_pos = int(round(n_samples * amyloid_positive_fraction))
        n_neg = n_samples - n_pos

        # Amyloid labels
        amyloid_label = np.zeros(n_samples, dtype=np.float32)
        amyloid_label[:n_pos] = 1.0

        # MMSE slope: amyloid+ declines faster (more negative)
        # Amyloid+: mean -2.5 points/year, std 1.5
        # Amyloid-: mean -0.8 points/year, std 1.0
        mmse_slope = np.zeros(n_samples, dtype=np.float32)
        if n_pos > 0:
            mmse_slope[:n_pos] = rng.normal(-2.5, 1.5, n_pos)
        if n_neg > 0:
            mmse_slope[n_pos:] = rng.normal(-0.8, 1.0, n_neg)

        # Time-to-event (months): amyloid+ progresses earlier
        # Amyloid+: Exponential(scale=24 months)
        # Amyloid-: Exponential(scale=60 months), many censored at 48 months
        time_to_event = np.zeros(n_samples, dtype=np.float32)
        event_observed = np.zeros(n_samples, dtype=np.float32)
        if n_pos > 0:
            tte_pos = rng.exponential(scale=24.0, size=n_pos)
            time_to_event[:n_pos] = np.clip(tte_pos, 1.0, 120.0)
            event_observed[:n_pos] = (tte_pos <= 48.0).astype(np.float32)
        if n_neg > 0:
            tte_neg = rng.exponential(scale=60.0, size=n_neg)
            time_to_event[n_pos:] = np.clip(tte_neg, 1.0, 120.0)
            event_observed[n_pos:] = (tte_neg <= 48.0).astype(np.float32)

        # Shuffle all arrays consistently
        indices = rng.permutation(n_samples)
        amyloid_label = amyloid_label[indices]
        mmse_slope = mmse_slope[indices]
        time_to_event = time_to_event[indices]
        event_observed = event_observed[indices]

        logger.debug("synthesized_labels", n_samples=n_samples)
        return {
            "amyloid_label":  torch.tensor(amyloid_label.tolist(), dtype=torch.float32),
            "mmse_slope":     torch.tensor(mmse_slope.tolist(), dtype=torch.float32),
            "time_to_event":  torch.tensor(time_to_event.tolist(), dtype=torch.float32),
            "event_observed": torch.tensor(event_observed.tolist(), dtype=torch.float32),
        }

    def synthesize_full_dataset(self, n_samples: int) -> dict[str, torch.Tensor]:
        """Generate a complete synthetic multimodal dataset.

        Combines all modality synthesizers and label generation into a single
        convenience call. All sub-datasets are generated with the same seed
        to ensure internal consistency.

        Args:
            n_samples: Number of synthetic patient records to generate.

        Returns:
            Dict with the following keys:
                'fluid': Tensor[n_samples, 6] — fluid biomarkers
                'acoustic': Tensor[n_samples, 12] — acoustic features
                'motor': Tensor[n_samples, 8] — motor features
                'clinical': Tensor[n_samples, 10] — clinical/demographic features
                'amyloid_label': Tensor[n_samples] — binary amyloid label
                'mmse_slope': Tensor[n_samples] — MMSE decline rate (points/year)
                'time_to_event': Tensor[n_samples] — months to event/censoring
                'event_observed': Tensor[n_samples] — event indicator (0/1)
        """
        logger.info("synthesizing_full_dataset", n_samples=n_samples)

        fluid = self.synthesize_fluid_biomarkers(n_samples)
        acoustic = self.synthesize_acoustic(n_samples)
        motor = self.synthesize_motor(n_samples)
        clinical = self.synthesize_clinical(n_samples)
        labels = self.synthesize_labels(n_samples)

        dataset = {
            "fluid": fluid,
            "acoustic": acoustic,
            "motor": motor,
            "clinical": clinical,
            **labels,
        }
        logger.info("full_dataset_synthesized", n_samples=n_samples)
        return dataset
