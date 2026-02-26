"""NeuroFusion-AD data pipeline package.

Exports the core data pipeline components for use by model training, API,
and testing infrastructure.

Modules:
    validators: InputValidator — clinical range enforcement
    adni_preprocessing: ADNIPreprocessor — normalization and imputation
    digital_biomarker_synthesis: DigitalBiomarkerSynthesizer — synthetic data generation
    dataset: NeuroFusionDataset, create_dataloaders, generate_synthetic_adni
"""

from src.data.validators import InputValidator
from src.data.adni_preprocessing import ADNIPreprocessor
from src.data.digital_biomarker_synthesis import DigitalBiomarkerSynthesizer
from src.data.dataset import (
    NeuroFusionDataset,
    create_dataloaders,
    generate_synthetic_adni,
)

__all__ = [
    "InputValidator",
    "ADNIPreprocessor",
    "DigitalBiomarkerSynthesizer",
    "NeuroFusionDataset",
    "create_dataloaders",
    "generate_synthetic_adni",
]
