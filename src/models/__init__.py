"""NeuroFusion-AD model package.

Exports all four modality encoders, the cross-modal attention fusion module,
the patient similarity GNN, and the full integrated NeuroFusionAD model
for direct import by consumers.

Example:
    >>> from src.models import (
    ...     FluidBiomarkerEncoder,
    ...     DigitalAcousticEncoder,
    ...     DigitalMotorEncoder,
    ...     ClinicalDemographicEncoder,
    ...     CrossModalAttention,
    ...     NeuroFusionGNN,
    ...     construct_patient_similarity_graph,
    ...     NeuroFusionAD,
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
from src.models.neurofusion_model import NeuroFusionAD, CLINICAL_DISCLAIMER

__all__ = [
    "FluidBiomarkerEncoder",
    "DigitalAcousticEncoder",
    "DigitalMotorEncoder",
    "ClinicalDemographicEncoder",
    "CrossModalAttention",
    "NeuroFusionGNN",
    "construct_patient_similarity_graph",
    "NeuroFusionAD",
    "CLINICAL_DISCLAIMER",
]
