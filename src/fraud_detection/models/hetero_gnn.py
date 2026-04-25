"""End-to-end heterogeneous fraud-detection GNN.

Architecture (implementation plan, page 7)::

    Input projections (per node type -> 128d)
        -> 3x HeteroGNNLayer (SAGEConv on tx-entity, GATConv-4head on card-card,
                              residual + LayerNorm + ELU + Dropout(0.3))
        -> Embedding head (128 -> 64d) on the transaction node type
        -> Classification head (64 -> 32 -> 1)
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch_geometric.data import HeteroData

from fraud_detection.models.gnn_layers import HeteroGNNLayer
from fraud_detection.utils.logging import get_logger

log = get_logger(__name__)


class FraudHeteroGNN(nn.Module):
    """Stacked heterogeneous GNN classifier with extractable 64-d embeddings.

    Parameters
    ----------
    node_feature_dims
        ``{node_type: input_feature_dim}`` -- typically ``{nt: data[nt].num_node_features
        for nt in data.node_types}``.
    edge_types
        All edge triples present in the data (forward + reverse). Pass
        ``data.edge_types`` after ``T.ToUndirected()`` so reverse relations
        are present too.
    hidden_dim
        Hidden dim of every HeteroGNNLayer. Plan default: 128.
    embedding_dim
        Dim of the per-transaction embedding emitted by ``get_embeddings``.
        This is what the GNN+XGBoost ensemble consumes. Plan default: 64.
    n_layers
        How many ``HeteroGNNLayer``s to stack. Plan default: 3.
    heads
        Heads per GAT layer. Plan default: 4.
    dropout
        Dropout for the GNN layers and the classifier. Plan default: 0.3.
    classifier_hidden
        Hidden dim of the classifier MLP. Plan default: 32 (i.e. 64 -> 32 -> 1).
    target_node_type
        Which node type the classifier acts on. The IEEE-CIS task is
        per-transaction so this is ``"transaction"``.
    """

    def __init__(
        self,
        *,
        node_feature_dims: dict[str, int],
        edge_types: Iterable[tuple[str, str, str]],
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        n_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
        classifier_hidden: int = 32,
        target_node_type: str = "transaction",
    ) -> None:
        super().__init__()
        if target_node_type not in node_feature_dims:
            msg = (
                f"target_node_type '{target_node_type}' not in node_feature_dims "
                f"keys: {sorted(node_feature_dims)}"
            )
            raise ValueError(msg)
        if n_layers < 1:
            msg = f"n_layers must be >= 1 (got {n_layers})"
            raise ValueError(msg)

        self.node_types = list(node_feature_dims)
        self.edge_types = list(edge_types)
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.dropout_p = dropout
        self.target_node_type = target_node_type

        # 1. Per-node-type input projection -> hidden_dim
        self.input_proj = nn.ModuleDict(
            {
                nt: nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ELU(inplace=True),
                )
                for nt, in_dim in node_feature_dims.items()
            }
        )

        # 2. Stack of HeteroGNNLayer
        self.layers = nn.ModuleList(
            [
                HeteroGNNLayer(
                    edge_types=self.edge_types,
                    node_types=self.node_types,
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    heads=heads,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # 3. Embedding head: hidden_dim (128) -> embedding_dim (64)
        self.embedding_head = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ELU(inplace=True),
        )

        # 4. Classification head: 64 -> 32 -> 1
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
        )

    # ------------------------------------------------------------------
    # forward / get_embeddings / get_logits
    # ------------------------------------------------------------------

    def encode(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Run input projection + GNN stack. Returns hidden-dim node features."""
        # Input projection (handles different per-type input dims).
        h = {nt: self.input_proj[nt](x) for nt, x in x_dict.items() if nt in self.input_proj}
        for layer in self.layers:
            h = layer(h, edge_index_dict)
        return h

    def get_embeddings(
        self,
        data: HeteroData,
        *,
        target_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the 64-d embedding vector per (selected) target node.

        Used by the ensemble to feed XGBoost. If ``target_indices`` is
        ``None``, returns embeddings for **all** target-type nodes.
        """
        h = self.encode(data.x_dict, data.edge_index_dict)
        target_h = self.embedding_head(h[self.target_node_type])
        if target_indices is not None:
            target_h = target_h[target_indices]
        return target_h

    def forward(
        self,
        data: HeteroData,
        *,
        target_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return raw logits (not sigmoid) per target node.

        Pair with :class:`fraud_detection.models.losses.FocalLoss` which
        operates on logits.
        """
        emb = self.get_embeddings(data, target_indices=target_indices)
        logits = self.classifier(emb).squeeze(-1)
        return logits

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"node_types={self.node_types}, hidden={self.hidden_dim}, "
            f"emb={self.embedding_dim}, n_layers={self.n_layers}, "
            f"target={self.target_node_type!r}, "
            f"params={self.n_parameters():,}"
        )


__all__: list[str] = ["FraudHeteroGNN"]
