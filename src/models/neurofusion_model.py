"""NeuroFusion-AD full integrated model.

Combines 4 modality encoders + CrossModalAttention fusion + GraphSAGE GNN
with 3 output heads for Alzheimer's disease progression prediction.

Architecture (SAD-001 § 5.4):
    1. Encode each modality: 4 encoders -> [batch, 768] each
    2. Fuse: CrossModalAttention -> [batch, 768]
    3. Build patient similarity graph from fused embeddings
    4. GNN aggregation: NeuroFusionGNN -> [batch, 768]
    5. Three output heads:
       - Classification: Linear(768, 1) -> sigmoid (amyloid positivity)
       - Regression:     Linear(768, 1)           (MMSE slope, points/year)
       - Survival:       Linear(768, 1)            (Cox log-hazard)

Loss functions:
    - Classification: BCEWithLogitsLoss (note: head outputs logit, not sigmoid)
    - Regression: MSELoss
    - Survival: Cox partial likelihood (implemented as a utility function)

Mandatory disclaimer on every forward pass output:
    "This tool is intended to support, not replace, clinical judgment."

Document traceability:
    SAD-001 § 5.4 — Full Model Architecture
    SRS-001 § 3.1 — System Overview
    RMF-001 § 4.1 — Risk Controls
"""

from __future__ import annotations

import torch
import torch.nn as nn
import structlog

from src.models.encoders import (
    FluidBiomarkerEncoder,
    DigitalAcousticEncoder,
    DigitalMotorEncoder,
    ClinicalDemographicEncoder,
)
from src.models.cross_modal_attention import CrossModalAttention
from src.models.gnn import NeuroFusionGNN, construct_patient_similarity_graph

log = structlog.get_logger(__name__)

CLINICAL_DISCLAIMER = (
    "This tool is intended to support, not replace, clinical judgment."
)


class NeuroFusionAD(nn.Module):
    """Full NeuroFusion-AD model for Alzheimer's disease progression prediction.

    This SaMD (Software as a Medical Device) tool accepts multimodal patient
    data and produces three complementary risk assessments:

    1. Amyloid positivity probability (classification)
    2. MMSE slope (cognitive decline rate, regression)
    3. Cox log-hazard for disease progression (survival analysis)

    All outputs are accompanied by the mandatory clinical disclaimer.
    This tool is NOT a diagnostic device — outputs are decision support only.

    Attributes:
        fluid_encoder: FluidBiomarkerEncoder (6 -> 768).
        acoustic_encoder: DigitalAcousticEncoder (12 -> 768).
        motor_encoder: DigitalMotorEncoder (8 -> 768).
        clinical_encoder: ClinicalDemographicEncoder (10 -> 768).
        fusion: CrossModalAttention (768, num_heads=8).
        gnn: NeuroFusionGNN (3-layer GraphSAGE, hidden_dim=768).
        classifier: Linear(768, 1) — amyloid logit.
        regressor: Linear(768, 1) — MMSE slope.
        survival_head: Linear(768, 1) — Cox log-hazard.
        graph_threshold: Cosine similarity threshold for graph construction (0.7).

    Args:
        embed_dim: Embedding dimension (default 768).
        num_heads: Attention heads (default 8).
        graph_threshold: Similarity threshold for patient graph (default 0.7).
        dropout: Dropout probability (default 0.1).

    Example:
        >>> model = NeuroFusionAD()
        >>> batch = {
        ...     'fluid':    torch.zeros(16, 6),
        ...     'acoustic': torch.zeros(16, 12),
        ...     'motor':    torch.randn(16, 8),
        ...     'clinical': torch.zeros(16, 10),
        ... }
        >>> batch['clinical'][:, 3] = 25.0
        >>> batch['fluid'][:, 0] = 5.0
        >>> batch['fluid'][:, 1] = 0.1
        >>> batch['fluid'][:, 2] = 50.0
        >>> batch['acoustic'][:, 0] = 0.005
        >>> batch['acoustic'][:, 1] = 0.05
        >>> outputs = model(batch)
        >>> outputs['amyloid_logit'].shape
        torch.Size([16, 1])
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        graph_threshold: float = 0.7,
        dropout: float = 0.4,
    ) -> None:
        """Initialise NeuroFusionAD.

        Phase 2B defaults: embed_dim=256, num_heads=4, dropout=0.4.

        Args:
            embed_dim: Embedding dimension (default 256).
            num_heads: Attention heads in CrossModalAttention (default 4).
            graph_threshold: Cosine similarity threshold for patient graph (default 0.7).
            dropout: Dropout probability (default 0.4).
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.graph_threshold = graph_threshold

        # Step 1 — Modality encoders (SAD-001 § 5.1)
        self.fluid_encoder = FluidBiomarkerEncoder()
        self.acoustic_encoder = DigitalAcousticEncoder()
        self.motor_encoder = DigitalMotorEncoder()
        self.clinical_encoder = ClinicalDemographicEncoder()

        # Step 2 — Cross-modal attention fusion (SAD-001 § 5.2)
        self.fusion = CrossModalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Step 3 — Patient similarity GNN (SAD-001 § 5.3)
        self.gnn = NeuroFusionGNN(
            hidden_dim=embed_dim,
            dropout=dropout,
        )

        # Step 4 — Output heads (SAD-001 § 5.4)
        # Classification head: outputs raw logit; BCEWithLogitsLoss applied externally
        self.classifier = nn.Linear(embed_dim, 1)
        # Regression head: predicted MMSE slope (points/year)
        self.regressor = nn.Linear(embed_dim, 1)
        # Survival head: Cox partial likelihood log-hazard
        self.survival_head = nn.Linear(embed_dim, 1)

        log.info(
            "NeuroFusionAD initialised",
            embed_dim=embed_dim,
            num_heads=num_heads,
            graph_threshold=graph_threshold,
            dropout=dropout,
            total_params=self.count_parameters(),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run full forward pass for a batch of patients.

        Processing pipeline:
            1. Encode each modality with its dedicated encoder
            2. Fuse embeddings with CrossModalAttention
            3. Build patient similarity graph from fused embeddings
            4. Aggregate neighborhood context with NeuroFusionGNN
            5. Produce three output head predictions

        Args:
            batch: Dict with keys:
                - 'fluid':    FloatTensor [batch, 6]
                - 'acoustic': FloatTensor [batch, 12]
                - 'motor':    FloatTensor [batch, 8]
                - 'clinical': FloatTensor [batch, 10]

        Returns:
            Dict with keys:
                - 'amyloid_logit':   FloatTensor [batch, 1] — BCEWithLogitsLoss compatible
                - 'mmse_slope':      FloatTensor [batch, 1] — predicted points/year
                - 'cox_log_hazard':  FloatTensor [batch, 1] — for Cox partial likelihood
                - 'fused_embedding': FloatTensor [batch, 768] — post-GNN embedding (for SHAP)
                - 'disclaimer':      str — mandatory clinical disclaimer

        Raises:
            KeyError: If required keys are missing from batch.
            ValueError: If encoder input validation fails.
        """
        # Validate that all required modality keys are present
        required_keys = ("fluid", "acoustic", "motor", "clinical")
        for key in required_keys:
            if key not in batch:
                raise KeyError(
                    f"Required key '{key}' is missing from batch. "
                    f"Batch must contain: {required_keys}."
                )

        batch_size = batch["fluid"].shape[0]

        log.debug(
            "NeuroFusionAD forward start",
            batch_size=batch_size,
        )

        # --- Step 1: Encode each modality ---
        fluid_emb: torch.Tensor = self.fluid_encoder(batch["fluid"])       # [B, 768]
        acoustic_emb: torch.Tensor = self.acoustic_encoder(batch["acoustic"])  # [B, 768]
        motor_emb: torch.Tensor = self.motor_encoder(batch["motor"])       # [B, 768]
        clinical_emb: torch.Tensor = self.clinical_encoder(batch["clinical"])  # [B, 768]

        # --- Step 2: Cross-modal attention fusion ---
        # fluid placed at position 0 as query anchor (SAD-001 § 5.2.1)
        fused: torch.Tensor = self.fusion(
            fluid_emb, acoustic_emb, motor_emb, clinical_emb
        )  # [B, 768]

        # --- Step 3: Build patient similarity graph from fused embeddings ---
        edge_index, _edge_weight = construct_patient_similarity_graph(
            fused, threshold=self.graph_threshold
        )  # edge_index: [2, num_edges]

        # --- Step 4: GNN neighbourhood aggregation ---
        gnn_out: torch.Tensor = self.gnn(fused, edge_index)  # [B, 768]

        # --- Step 5: Output heads ---
        amyloid_logit: torch.Tensor = self.classifier(gnn_out)   # [B, 1]
        mmse_slope: torch.Tensor = self.regressor(gnn_out)        # [B, 1]
        cox_log_hazard: torch.Tensor = self.survival_head(gnn_out)  # [B, 1]

        log.debug(
            "NeuroFusionAD forward complete",
            batch_size=batch_size,
            amyloid_logit_shape=list(amyloid_logit.shape),
            mmse_slope_shape=list(mmse_slope.shape),
            cox_log_hazard_shape=list(cox_log_hazard.shape),
        )

        return {
            "amyloid_logit": amyloid_logit,
            "mmse_slope": mmse_slope,
            "cox_log_hazard": cox_log_hazard,
            "fused_embedding": gnn_out,
            "disclaimer": CLINICAL_DISCLAIMER,
        }

    def count_parameters(self) -> int:
        """Count total trainable parameters.

        Returns:
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
