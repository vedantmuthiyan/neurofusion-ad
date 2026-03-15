"""Patient similarity graph and GNN for NeuroFusion-AD.

Constructs a cosine-similarity patient graph and runs GraphSAGE
message-passing to aggregate neighborhood context.

Architecture:
    - Graph construction: cosine similarity, threshold = 0.7
    - GNN: 3-layer GraphSAGE (SAGEConv), hidden_dim = 768
    - Residual connections between layers
    - LayerNorm after each layer

Document traceability:
    SAD-001 § 5.3 — Patient Similarity Graph
    SRS-001 § 4.4 — Graph-Based Patient Contextualization
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import structlog

log = structlog.get_logger(__name__)


def construct_patient_similarity_graph(
    embeddings: torch.Tensor,
    threshold: float = 0.7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct a patient similarity graph from fused embeddings.

    Computes pairwise cosine similarity between patient embeddings and
    creates edges between all pairs with similarity >= threshold.

    Args:
        embeddings: Fused patient embeddings [n_patients, 768].
        threshold: Cosine similarity threshold for edge creation (default 0.7).

    Returns:
        Tuple of (edge_index, edge_weight):
            - edge_index: LongTensor [2, num_edges] — COO format edge indices.
            - edge_weight: FloatTensor [num_edges] — cosine similarity weights.

    Note:
        Self-loops are excluded (a patient is not similar to themselves
        in this graph sense; they are aggregated over neighbors only).
    """
    n_patients = embeddings.shape[0]

    # Normalise embeddings to unit length for cosine similarity computation
    # Shape: [n_patients, embed_dim]
    normed = F.normalize(embeddings, p=2, dim=1)

    # Pairwise cosine similarity matrix: [n_patients, n_patients]
    similarity_matrix = torch.mm(normed, normed.t())

    # Build edge mask: similarity >= threshold AND not a self-loop
    row_idx = torch.arange(n_patients, device=embeddings.device)
    self_loop_mask = row_idx.unsqueeze(0) == row_idx.unsqueeze(1)  # [n, n] bool

    edge_mask = (similarity_matrix >= threshold) & (~self_loop_mask)

    # Extract non-zero indices -> COO format edge_index [2, num_edges]
    src, dst = edge_mask.nonzero(as_tuple=True)
    edge_index = torch.stack([src, dst], dim=0).long()

    # Extract corresponding cosine similarity weights
    edge_weight = similarity_matrix[src, dst]

    log.debug(
        "construct_patient_similarity_graph complete",
        n_patients=n_patients,
        threshold=threshold,
        num_edges=edge_index.shape[1],
    )
    return edge_index, edge_weight


class NeuroFusionGNN(nn.Module):
    """3-layer GraphSAGE GNN for patient-level context aggregation.

    After cross-modal fusion, each patient has a 768-dim embedding.
    This GNN performs 3 rounds of neighborhood aggregation over the
    patient similarity graph to produce context-aware embeddings.

    Architecture:
        Layer 1: SAGEConv(768, 768) -> LayerNorm(768) -> GELU -> Dropout(0.1) + residual
        Layer 2: SAGEConv(768, 768) -> LayerNorm(768) -> GELU -> Dropout(0.1) + residual
        Layer 3: SAGEConv(768, 768) -> LayerNorm(768) [no dropout on final layer]

    Attributes:
        hidden_dim: Feature dimension (768 throughout).
        conv1, conv2, conv3: SAGEConv layers.
        norm1, norm2, norm3: LayerNorm layers.
        dropout: Dropout(0.1).

    Args:
        hidden_dim: Hidden dimension (default 768).
        dropout: Dropout probability (default 0.1).

    Example:
        >>> gnn = NeuroFusionGNN()
        >>> x = torch.randn(16, 768)
        >>> # Construct graph for a batch of 16 patients
        >>> edge_index, _ = construct_patient_similarity_graph(x, threshold=0.7)
        >>> out = gnn(x, edge_index)
        >>> out.shape
        torch.Size([16, 768])
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.4,
    ) -> None:
        """Initialise NeuroFusionGNN.

        Phase 2B: 2-layer (was 3), hidden_dim=256 (was 768), dropout=0.4.

        Args:
            hidden_dim: Node feature dimension throughout the network (default 256).
            dropout: Dropout probability (default 0.4).
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Two GraphSAGE convolution layers (Phase 2B — reduced capacity)
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        # LayerNorm after each convolution
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Shared dropout
        self.dropout = nn.Dropout(p=dropout)

        log.info(
            "NeuroFusionGNN initialised",
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=2,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Run 2 rounds of GraphSAGE aggregation.

        Each layer applies SAGEConv -> LayerNorm -> GELU -> Dropout with
        a residual connection.

        Args:
            x: Node feature matrix [n_nodes, hidden_dim].
            edge_index: Edge indices [2, num_edges] in COO format.

        Returns:
            Updated node features [n_nodes, hidden_dim] after 2-layer aggregation.

        Raises:
            ValueError: If x does not have shape [..., hidden_dim].
        """
        if x.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Input x must have last dimension {self.hidden_dim}, "
                f"got {x.shape[-1]}. Full shape: {list(x.shape)}."
            )

        # Layer 1: SAGEConv + LayerNorm + GELU + Dropout + residual
        h1 = self.conv1(x, edge_index)
        h1 = self.norm1(h1)
        h1 = F.gelu(h1)
        h1 = self.dropout(h1)
        x = x + h1  # residual connection

        # Layer 2: SAGEConv + LayerNorm + GELU + Dropout + residual
        h2 = self.conv2(x, edge_index)
        h2 = self.norm2(h2)
        h2 = F.gelu(h2)
        h2 = self.dropout(h2)
        x = x + h2  # residual connection

        log.debug(
            "NeuroFusionGNN forward complete",
            n_nodes=x.shape[0],
            output_shape=list(x.shape),
        )
        return x
