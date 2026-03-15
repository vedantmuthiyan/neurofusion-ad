"""Unit tests for the patient similarity graph constructor and NeuroFusionGNN.

Tests are self-contained — only torch is used for synthetic tensor generation.
No real patient data (PHI) is used in any test.

Test coverage:
    test_construct_graph_shape      : edge_index has shape [2, num_edges]
    test_construct_graph_no_self_loops : no edge from node i to itself
    test_construct_graph_threshold  : higher threshold produces fewer edges
    test_construct_graph_all_ones   : identical embeddings -> all non-self pairs connected
    test_gnn_output_shape           : GNN output shape is [n_nodes, 768]
    test_gnn_output_type            : GNN output dtype is torch.float32
    test_gnn_no_nan_output          : GNN output contains no NaN values
    test_gnn_isolated_nodes         : empty edge_index produces valid output

Document traceability:
    SAD-001 § 5.3 — Patient Similarity Graph
    SRS-001 § 4.4 — Graph-Based Patient Contextualization
    IEC 62304 § 5.5.5 — Software unit verification
"""

from __future__ import annotations

import pytest
import torch

from src.models.gnn import NeuroFusionGNN, construct_patient_similarity_graph

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HIDDEN_DIM = 256  # Phase 2B: reduced from 768 to 256
N_PATIENTS = 16


# ---------------------------------------------------------------------------
# Shared helpers — pure torch, no numpy
# ---------------------------------------------------------------------------

def _make_node_features(
    n_nodes: int = N_PATIENTS,
    seed: int = 0,
) -> torch.Tensor:
    """Return random float32 node features of shape [n_nodes, 768].

    Args:
        n_nodes: Number of nodes (patients).
        seed: Manual seed for reproducibility.

    Returns:
        Float32 tensor of shape [n_nodes, 768].
    """
    torch.manual_seed(seed)
    return torch.randn(n_nodes, HIDDEN_DIM, dtype=torch.float32)


# ===========================================================================
# construct_patient_similarity_graph tests
# ===========================================================================

class TestConstructPatientSimilarityGraph:
    """Unit tests for construct_patient_similarity_graph (SAD-001 § 5.3)."""

    def test_construct_graph_shape(self) -> None:
        """edge_index must have shape [2, num_edges] and edge_weight [num_edges].

        Verifies that the COO format is correct: first row contains source
        indices, second row contains destination indices.
        """
        x = _make_node_features(N_PATIENTS)
        edge_index, edge_weight = construct_patient_similarity_graph(x, threshold=0.7)

        assert edge_index.dim() == 2, (
            f"edge_index must be 2-dimensional, got {edge_index.dim()}D."
        )
        assert edge_index.shape[0] == 2, (
            f"edge_index first dimension must be 2 (src/dst), got {edge_index.shape[0]}."
        )
        num_edges = edge_index.shape[1]
        assert edge_weight.shape == (num_edges,), (
            f"edge_weight shape {edge_weight.shape} must match [num_edges] = [{num_edges}]."
        )

    def test_construct_graph_no_self_loops(self) -> None:
        """No edge from node i to itself must be present in the graph.

        Self-loops are explicitly excluded per SAD-001 § 5.3.1 to ensure
        patients are only contextualised by their neighbors.
        """
        x = _make_node_features(N_PATIENTS)
        edge_index, _ = construct_patient_similarity_graph(x, threshold=0.0)

        # With threshold=0.0 all pairs with cosine similarity >= 0 are included
        # (excluding self-loops). Check that no src == dst edge exists.
        if edge_index.shape[1] > 0:
            src = edge_index[0]
            dst = edge_index[1]
            self_loops = (src == dst).any()
            assert not self_loops, (
                "construct_patient_similarity_graph produced self-loop edges."
            )

    def test_construct_graph_threshold(self) -> None:
        """A higher similarity threshold must produce fewer or equal edges.

        This validates the monotonicity property of the threshold filter:
        stricter thresholds should not add new edges.
        """
        x = _make_node_features(N_PATIENTS)

        _, ew_low = construct_patient_similarity_graph(x, threshold=0.3)
        ei_high, _ = construct_patient_similarity_graph(x, threshold=0.8)
        _, ew_low_ref = construct_patient_similarity_graph(x, threshold=0.3)

        num_edges_low = ew_low_ref.shape[0]
        num_edges_high = ei_high.shape[1]

        assert num_edges_high <= num_edges_low, (
            f"Higher threshold (0.8) produced {num_edges_high} edges, "
            f"which is more than lower threshold (0.3) with {num_edges_low} edges. "
            "Edge count must be monotonically non-increasing with threshold."
        )

    def test_construct_graph_all_ones(self) -> None:
        """Identical unit embeddings must produce cosine similarity = 1.0 for
        all pairs, so every non-self pair must be connected.

        Uses a threshold of 0.99 to confirm all n*(n-1) directed edges exist.
        """
        n = 8
        # All patients have the exact same embedding -> cosine sim = 1.0 for all pairs
        x = torch.ones(n, HIDDEN_DIM, dtype=torch.float32)

        edge_index, edge_weight = construct_patient_similarity_graph(x, threshold=0.99)

        expected_edges = n * (n - 1)  # directed graph, no self-loops
        assert edge_index.shape[1] == expected_edges, (
            f"Expected {expected_edges} edges for identical embeddings "
            f"(n={n}, directed, no self-loops), got {edge_index.shape[1]}."
        )
        # All weights should be ~1.0
        assert torch.allclose(
            edge_weight, torch.ones_like(edge_weight), atol=1e-5
        ), "Edge weights must be ~1.0 for identical unit embeddings."


# ===========================================================================
# NeuroFusionGNN tests
# ===========================================================================

class TestNeuroFusionGNN:
    """Unit tests for NeuroFusionGNN (SAD-001 § 5.3)."""

    @pytest.fixture
    def gnn(self) -> NeuroFusionGNN:
        """Provide a freshly initialised NeuroFusionGNN in eval mode.

        Returns:
            NeuroFusionGNN instance set to eval mode.
        """
        model = NeuroFusionGNN()
        model.eval()
        return model

    @pytest.fixture
    def default_graph(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Provide node features and edge_index for N_PATIENTS patients.

        Graph is constructed with threshold=0.5 to ensure a reasonable
        number of edges exist for a typical random embedding batch.

        Returns:
            Tuple of (x, edge_index) where x is [N_PATIENTS, 768] and
            edge_index is [2, num_edges].
        """
        x = _make_node_features(N_PATIENTS)
        edge_index, _ = construct_patient_similarity_graph(x, threshold=0.5)
        return x, edge_index

    def test_gnn_output_shape(
        self,
        gnn: NeuroFusionGNN,
        default_graph: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """GNN output must have shape [n_nodes, 768].

        Args:
            gnn: NeuroFusionGNN fixture.
            default_graph: (x, edge_index) fixture.
        """
        x, edge_index = default_graph
        with torch.no_grad():
            out = gnn(x, edge_index)
        assert out.shape == (N_PATIENTS, HIDDEN_DIM), (
            f"Expected shape ({N_PATIENTS}, {HIDDEN_DIM}), got {out.shape}"
        )

    def test_gnn_output_type(
        self,
        gnn: NeuroFusionGNN,
        default_graph: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """GNN output must be of dtype torch.float32.

        Args:
            gnn: NeuroFusionGNN fixture.
            default_graph: (x, edge_index) fixture.
        """
        x, edge_index = default_graph
        with torch.no_grad():
            out = gnn(x, edge_index)
        assert out.dtype == torch.float32, (
            f"Expected torch.float32, got {out.dtype}"
        )

    def test_gnn_no_nan_output(
        self,
        gnn: NeuroFusionGNN,
        default_graph: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """GNN output must contain no NaN values for valid random inputs.

        Args:
            gnn: NeuroFusionGNN fixture.
            default_graph: (x, edge_index) fixture.
        """
        x, edge_index = default_graph
        with torch.no_grad():
            out = gnn(x, edge_index)
        assert not torch.isnan(out).any(), (
            "NeuroFusionGNN produced NaN values for valid random inputs."
        )

    def test_gnn_isolated_nodes(self, gnn: NeuroFusionGNN) -> None:
        """GNN must produce valid output even when the graph has no edges.

        When edge_index is empty (shape [2, 0]), each node is isolated and
        the SAGEConv layers fall back to self-aggregation only. The output
        must be finite and have the correct shape.

        Args:
            gnn: NeuroFusionGNN fixture.
        """
        x = _make_node_features(N_PATIENTS)
        # Explicitly create an empty edge_index in COO format
        edge_index = torch.zeros(2, 0, dtype=torch.long)

        with torch.no_grad():
            out = gnn(x, edge_index)

        assert out.shape == (N_PATIENTS, HIDDEN_DIM), (
            f"Expected shape ({N_PATIENTS}, {HIDDEN_DIM}) for isolated nodes, "
            f"got {out.shape}"
        )
        assert not torch.isnan(out).any(), (
            "NeuroFusionGNN produced NaN values when graph has no edges."
        )
        assert torch.isfinite(out).all(), (
            "NeuroFusionGNN produced non-finite values when graph has no edges."
        )
