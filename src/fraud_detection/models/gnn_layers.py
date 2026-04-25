"""Heterogeneous GNN layers for the fraud-detection model.

Per the implementation plan (page 7) the architecture mixes two convolution
operators based on the relation between the two endpoints:

    transaction <-> entity      ->  SAGEConv  (cheap, scales to high degree)
    card <-> card               ->  GATConv (4 heads)  (attention reveals
                                              ring structure: shared_device
                                              and shared_address links)

This module exposes :class:`HeteroGNNLayer` -- one PyG ``HeteroConv`` block
plus the residual / LayerNorm / ELU / Dropout(0.3) wrapper. The
classification head and stacking lives in :mod:`hetero_gnn`.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv

# Edge types that use GATConv (entity-entity attention). All others use SAGEConv.
# Stored as ``(src, rel, dst)`` triples; ``rel`` is matched against this set so
# reverse edges added by :class:`torch_geometric.transforms.ToUndirected` (which
# produces ``rev_<rel>``) are picked up by the same dispatch.
GAT_RELATIONS: frozenset[str] = frozenset(
    {
        "shared_address",
        "shared_device",
        "rev_shared_address",
        "rev_shared_device",
    }
)


def _is_gat_edge(edge_type: tuple[str, str, str]) -> bool:
    src, rel, dst = edge_type
    # Card-card edges use GAT regardless of direction. We also gate on the
    # relation name so that any future entity-entity relation can be added
    # to GAT_RELATIONS without touching this dispatch.
    return rel in GAT_RELATIONS or (src == dst != "transaction")


class HeteroGNNLayer(nn.Module):
    """One GNN block: HeteroConv -> residual + LayerNorm + ELU + Dropout.

    Parameters
    ----------
    edge_types
        All edge triples present in the data (e.g. from ``data.edge_types``).
        Both forward and reverse types should be passed.
    in_dim
        Hidden dim of the input node embeddings (must be the same for every
        node type -- ensure with a per-type projection on the input).
    out_dim
        Hidden dim of the output. Plan default: 128.
    heads
        Number of attention heads in GATConv (entity-entity edges).
        Plan default: 4. The output is averaged over heads so the final
        dim stays ``out_dim``.
    dropout
        Dropout probability applied to attention weights (GAT only) and at
        the end of the block. Plan default: 0.3.
    aggr
        Cross-relation aggregation in HeteroConv (``"sum" | "mean" | "max"``).
        ``"sum"`` is the default and matches the canonical PyG examples.
    node_types
        All node types present in the data. Required to size the per-type
        LayerNorm modules.
    """

    def __init__(
        self,
        *,
        edge_types: Iterable[tuple[str, str, str]],
        node_types: Iterable[str],
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.3,
        aggr: str = "sum",
    ) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0:
            msg = f"in_dim and out_dim must be positive (got {in_dim}, {out_dim})"
            raise ValueError(msg)

        edge_types = list(edge_types)
        node_types = list(node_types)
        self.edge_types = edge_types
        self.node_types = node_types
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.dropout_p = dropout

        # Build the per-relation conv map.
        conv_map: dict[tuple[str, str, str], nn.Module] = {}
        for et in edge_types:
            if _is_gat_edge(et):
                # ``concat=False`` averages heads -> output dim = out_dim
                # (rather than heads * out_dim), which keeps the residual
                # path dim-compatible.
                conv_map[et] = GATConv(
                    in_dim,
                    out_dim,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                    add_self_loops=False,
                )
            else:
                conv_map[et] = SAGEConv((in_dim, in_dim), out_dim)
        self.conv = HeteroConv(conv_map, aggr=aggr)

        # Per-node-type LayerNorm + residual projection (when in_dim != out_dim).
        self.norms = nn.ModuleDict({nt: nn.LayerNorm(out_dim) for nt in node_types})
        if in_dim != out_dim:
            self.residual_projs = nn.ModuleDict(
                {nt: nn.Linear(in_dim, out_dim, bias=False) for nt in node_types}
            )
        else:
            self.residual_projs = None  # type: ignore[assignment]

        self.act = nn.ELU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Apply the layer to a hetero feature dict.

        Returns a new dict with the same keys but updated tensors.
        """
        # 1. Aggregate via HeteroConv
        out = self.conv(x_dict, edge_index_dict)

        # 2. Residual + norm + activation + dropout (per node type)
        for nt, h_in in x_dict.items():
            h_out = out.get(nt)
            if h_out is None:
                # Some node types may have no incoming edges in this layer's
                # relation map -- carry the input through unchanged but still
                # apply the norm so dims stay coherent for downstream layers.
                h_out = self.residual_projs[nt](h_in) if self.residual_projs is not None else h_in
            else:
                residual = (
                    self.residual_projs[nt](h_in) if self.residual_projs is not None else h_in
                )
                h_out = h_out + residual
            h_out = self.norms[nt](h_out)
            h_out = self.act(h_out)
            h_out = self.dropout(h_out)
            out[nt] = h_out
        return out

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        n_gat = sum(1 for et in self.edge_types if _is_gat_edge(et))
        n_sage = len(self.edge_types) - n_gat
        return (
            f"in={self.in_dim}, out={self.out_dim}, heads={self.heads}, "
            f"dropout={self.dropout_p}, edges={len(self.edge_types)} "
            f"(SAGE={n_sage}, GAT={n_gat})"
        )


__all__: list[str] = ["GAT_RELATIONS", "HeteroGNNLayer"]
