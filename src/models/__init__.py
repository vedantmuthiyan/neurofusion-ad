"""NeuroFusion-AD model package.

Exports all four modality encoders, the cross-modal attention fusion module,
and the patient similarity GNN for direct import by consumers.

Example:
    >>> from src.models import (
    ...     FluidBiomarkerEncoder,
    ...     DigitalAcousticEncoder,
    ...     DigitalMotorEncoder,
    ...     ClinicalDemographicEncoder,
    ...     CrossModalAttention,
    ...     NeuroFusionGNN,
    ...     construct_patient_similarity_graph,
    ... )
"""

from src.models.encoders import (
    ClinicalDemographicEncoder,
    DigitalAcousticEncoder,
    DigitalMotorEncoder,
    FluidBiomarkerEncoder,
)
from src.models.cross_modal_attention import CrossModalAttention
from src.models.gnn import NeuroFusionGNN, construct_patient_similarity_graph

__all__ = [
    "FluidBiomarkerEncoder",
    "DigitalAcousticEncoder",
    "DigitalMotorEncoder",
    "ClinicalDemographicEncoder",
    "CrossModalAttention",
    "NeuroFusionGNN",
    "construct_patient_similarity_graph",
]
