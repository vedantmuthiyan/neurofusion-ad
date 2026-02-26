"""PyTorch Dataset and DataLoader utilities for NeuroFusion-AD.

Provides NeuroFusionDataset (a PyTorch Dataset), a synthetic data factory
function, and a DataLoader creation utility with reproducible train/val/test
splits.

All patient IDs are stored as SHA-256 hashes to comply with IEC 62304
PHI handling requirements. No raw patient identifiers are ever stored or logged.

IEC 62304 Requirement Traceability: SRS-001 § 5.4 (Data Loading)
"""

import hashlib
import structlog
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from src.data.digital_biomarker_synthesis import DigitalBiomarkerSynthesizer

logger = structlog.get_logger(__name__)

# Required data keys for the dataset
_FEATURE_KEYS = ("fluid", "acoustic", "motor", "clinical")
_LABEL_KEYS = ("amyloid_label", "mmse_slope", "time_to_event", "event_observed")
_ALL_KEYS = _FEATURE_KEYS + _LABEL_KEYS

# Expected feature dimensions
_FEATURE_DIMS = {
    "fluid": 6,
    "acoustic": 12,
    "motor": 8,
    "clinical": 10,
}


def _hash_patient_id(patient_id: str) -> str:
    """Compute SHA-256 hash of a patient ID for PHI-safe logging.

    Args:
        patient_id: Raw patient identifier string.

    Returns:
        Hex-encoded SHA-256 hash of the patient ID.
    """
    return hashlib.sha256(patient_id.encode()).hexdigest()


class NeuroFusionDataset(Dataset):
    """PyTorch Dataset for NeuroFusion-AD multimodal data.

    Holds multimodal tensors and labels for a cohort of patients. All patient
    IDs are stored as SHA-256 hashes to ensure PHI compliance. Labels support
    all three output heads: classification (amyloid), regression (MMSE slope),
    and survival (time-to-event).

    Attributes:
        data: Dict of tensors keyed by modality and label name.
        patient_ids: Optional list of SHA-256-hashed patient ID strings.
        n_samples: Total number of patient records.

    IEC 62304 Traceability:
        SRS-001 § 5.4 — Dataset interface requirements
        SRS-001 § 6.1 — PHI handling requirements
    """

    def __init__(
        self,
        data: dict[str, torch.Tensor],
        patient_ids: list[str] | None = None,
    ) -> None:
        """Initialize NeuroFusionDataset.

        Args:
            data: Dict with the following keys and tensor shapes:
                'fluid': Tensor[N, 6] — fluid biomarker features
                'acoustic': Tensor[N, 12] — acoustic features
                'motor': Tensor[N, 8] — motor features
                'clinical': Tensor[N, 10] — clinical/demographic features
                'amyloid_label': Tensor[N] — binary amyloid positivity label
                'mmse_slope': Tensor[N] — MMSE decline rate (points/year)
                'time_to_event': Tensor[N] — months to progression/censoring
                'event_observed': Tensor[N] — event indicator (0=censored, 1=observed)
            patient_ids: Optional list of N patient ID strings. Strings are
                immediately SHA-256 hashed; the raw IDs are never stored.

        Raises:
            KeyError: If any required key is missing from data.
            ValueError: If tensors have mismatched first dimensions.
        """
        for key in _ALL_KEYS:
            if key not in data:
                raise KeyError(f"data dict is missing required key: '{key}'")

        # Verify all tensors have the same N
        n_samples = data["fluid"].shape[0]
        for key in _ALL_KEYS:
            if data[key].shape[0] != n_samples:
                raise ValueError(
                    f"Tensor '{key}' has {data[key].shape[0]} samples, "
                    f"expected {n_samples} (matching 'fluid')."
                )

        self.data = {key: data[key].float() for key in _ALL_KEYS}
        self.n_samples = n_samples

        # Hash patient IDs immediately — never store raw IDs
        if patient_ids is not None:
            if len(patient_ids) != n_samples:
                raise ValueError(
                    f"patient_ids has {len(patient_ids)} entries, "
                    f"expected {n_samples}."
                )
            self.patient_ids: list[str] | None = [
                _hash_patient_id(pid) for pid in patient_ids
            ]
        else:
            self.patient_ids = None

        logger.info("dataset_initialized", n_samples=n_samples)

    def __len__(self) -> int:
        """Return the number of patient records in the dataset.

        Returns:
            Integer count of samples.
        """
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return all features and labels for a single patient record.

        Args:
            idx: Integer index in [0, len(dataset)).

        Returns:
            Dict with the following keys and shapes:
                'fluid': Tensor[6]
                'acoustic': Tensor[12]
                'motor': Tensor[8]
                'clinical': Tensor[10]
                'amyloid_label': Tensor[] (scalar)
                'mmse_slope': Tensor[] (scalar)
                'time_to_event': Tensor[] (scalar)
                'event_observed': Tensor[] (scalar)

        Raises:
            IndexError: If idx is out of bounds.
        """
        if idx < 0 or idx >= self.n_samples:
            raise IndexError(
                f"Index {idx} is out of bounds for dataset with {self.n_samples} samples."
            )
        return {key: self.data[key][idx] for key in _ALL_KEYS}


def generate_synthetic_adni(n_samples: int = 200, seed: int = 42) -> "NeuroFusionDataset":
    """Generate a synthetic ADNI-like dataset for testing and development.

    Uses DigitalBiomarkerSynthesizer to create clinically plausible multimodal
    data. The output can be passed directly to create_dataloaders.

    WARNING: Synthetic data only. Not for clinical use. This function exists
    solely for testing and CI purposes.

    Args:
        n_samples: Number of synthetic patient records to generate.
            Recommended: 200 for unit tests, 1000+ for integration tests.
        seed: Random seed for reproducibility.

    Returns:
        NeuroFusionDataset ready for use with PyTorch DataLoader.
    """
    logger.info("generating_synthetic_adni", n_samples=n_samples, seed=seed)
    synthesizer = DigitalBiomarkerSynthesizer(seed=seed)
    data = synthesizer.synthesize_full_dataset(n_samples=n_samples)
    dataset = NeuroFusionDataset(data=data)
    logger.info("synthetic_adni_generated", n_samples=n_samples)
    return dataset


def create_dataloaders(
    dataset: "NeuroFusionDataset",
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    batch_size: int = 16,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Split a NeuroFusionDataset and create train/val/test DataLoaders.

    Performs a reproducible random split and returns three DataLoaders.
    The test fraction is automatically computed as 1 - train_fraction - val_fraction.
    The training DataLoader shuffles data; validation and test loaders do not.

    Args:
        dataset: NeuroFusionDataset to split.
        train_fraction: Fraction of data for training. Must be in (0, 1).
        val_fraction: Fraction of data for validation. Must be in (0, 1).
            train_fraction + val_fraction must be < 1.0.
        batch_size: Batch size for all three DataLoaders.
        seed: Random seed for the reproducible dataset split.

    Returns:
        Tuple of (train_loader, val_loader, test_loader) DataLoaders.

    Raises:
        ValueError: If fractions are invalid or their sum >= 1.0.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}.")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}.")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError(
            f"train_fraction ({train_fraction}) + val_fraction ({val_fraction}) "
            f"must be < 1.0 to leave room for a test split."
        )

    n_total = len(dataset)
    n_train = int(n_total * train_fraction)
    n_val = int(n_total * val_fraction)
    n_test = n_total - n_train - n_val

    if n_test < 1:
        raise ValueError(
            f"Dataset of {n_total} samples with train={train_fraction} and "
            f"val={val_fraction} leaves no samples for test split."
        )

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    logger.info(
        "dataloaders_created",
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        batch_size=batch_size,
    )
    return train_loader, val_loader, test_loader
