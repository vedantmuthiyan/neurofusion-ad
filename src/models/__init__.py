"""NeuroFusion-AD model package.

Exports all four modality encoders for direct import by consumers.

Example:
    >>> from src.models import (
    ...     FluidBiomarkerEncoder,
    ...     DigitalAcousticEncoder,
    ...     DigitalMotorEncoder,
    ...     ClinicalDemographicEncoder,
    ... )
"""

from src.models.encoders import (
    ClinicalDemographicEncoder,
    DigitalAcousticEncoder,
    DigitalMotorEncoder,
    FluidBiomarkerEncoder,
)

__all__ = [
    "FluidBiomarkerEncoder",
    "DigitalAcousticEncoder",
    "DigitalMotorEncoder",
    "ClinicalDemographicEncoder",
]
