"""Build a heterogeneous transaction graph for the IEEE-CIS dataset.

Implements the topology specified in Phase 2 of the plan:

* **7 node types**: transaction, card, address, email, device, ip_address,
  merchant (V-feature cluster).
* **8 edge types**: 6 transaction -> entity edges and 2 card <-> card
  entity-entity edges (shared_address, shared_device).

The output is a single :class:`torch_geometric.data.HeteroData` with:

* Per-node features on ``data[node_type].x`` (float32).
* Edge indices on ``data[src, relation, dst].edge_index`` (int64, [2, E]).
* Transaction labels on ``data['transaction'].y`` (int8).
* Three boolean masks on ``data['transaction'].{train,val,test}_mask``
  derived from the temporal splits.

The builder is stateful so we can save and re-load the learned node
indices + K-means centers for use at serve time.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from torch_geometric.data import HeteroData

from fraud_detection.utils.config import AppConfig, load_config
from fraud_detection.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Schema constants (the central source of truth for Phase 2)
# ---------------------------------------------------------------------------

NODE_TYPES: tuple[str, ...] = (
    "transaction",
    "card",
    "address",
    "email",
    "device",
    "ip_address",
    "merchant",
)

EDGE_TYPES: tuple[tuple[str, str, str], ...] = (
    ("transaction", "uses_card", "card"),
    ("transaction", "from_address", "address"),
    ("transaction", "from_email", "email"),
    ("transaction", "from_device", "device"),
    ("transaction", "from_ip", "ip_address"),
    ("transaction", "at_merchant", "merchant"),
    ("card", "shared_address", "card"),
    ("card", "shared_device", "card"),
)

# Per-node feature dimensions (from Phase 2 spec, page 5).
NODE_FEATURE_DIMS: dict[str, int] = {
    "transaction": 50,
    "card": 8,
    "address": 4,
    "email": 3,
    "device": 4,
    "ip_address": 6,
    "merchant": 10,
}

# Known free-email providers -> is_free_provider feature.
_FREE_EMAIL_DOMAINS: frozenset[str] = frozenset(
    {
        "gmail.com",
        "yahoo.com",
        "hotmail.com",
        "outlook.com",
        "aol.com",
        "icloud.com",
        "mail.com",
        "protonmail.com",
        "live.com",
        "msn.com",
        "me.com",
        "yandex.ru",
    }
)


@dataclass
class GraphBuilderState:
    """Everything learned during :meth:`HeteroGraphBuilder.fit` that must be
    replayed at serve time for online scoring."""

    # Map node_type -> {entity_key: node_index}. "transaction" uses the
    # TransactionID as key.
    node_index_maps: dict[str, dict[Any, int]] = field(default_factory=dict)
    # Per-node-type feature matrices keyed by node_type at fit-time row order.
    # Persisted so the serve path can append new entities without having to
    # re-compute everyone's features.
    entity_feature_frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    # K-means model for merchant clustering (V307-V316).
    merchant_kmeans: MiniBatchKMeans | None = None
    # IP-address bin edges (for consistent binning at serve time).
    id01_bins: np.ndarray | None = None
    id02_bins: np.ndarray | None = None

    def to_file(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_file(cls, path: str | Path) -> GraphBuilderState:
        with Path(path).open("rb") as f:
            state: GraphBuilderState = pickle.load(f)
        return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quantile_bin_edges(series: pd.Series, n_bins: int = 10) -> np.ndarray:
    """Return bin edges for quantile bucketing; robust to mostly-missing series."""
    clean = series.dropna()
    if len(clean) < 2:
        return np.array([0.0, 1.0])
    try:
        edges = np.quantile(clean.to_numpy(), np.linspace(0, 1, n_bins + 1))
    except (ValueError, IndexError):  # pragma: no cover -- defensive
        return np.array([clean.min(), clean.max()])
    # Guarantee strictly-increasing edges.
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.array([edges[0], edges[0] + 1])
    return edges


def _apply_bin(values: pd.Series, edges: np.ndarray) -> np.ndarray:
    """Map numeric values into bin indices [0, len(edges)-2] with NaN -> -1."""
    out = np.full(len(values), -1, dtype=np.int32)
    mask = values.notna().to_numpy()
    if mask.any() and len(edges) >= 2:
        binned = np.clip(
            np.searchsorted(edges, values.to_numpy()[mask], side="right") - 1,
            0,
            len(edges) - 2,
        )
        out[mask] = binned
    return out


def _select_transaction_feature_cols(df: pd.DataFrame) -> list[str]:
    """Pick up to 50 **numeric** columns for the transaction node feature tensor.

    Strategy: prefer the V-columns with highest absolute correlation to
    ``isFraud`` (or variance if target unavailable), then pad with
    TransactionAmt, log1p amount, TransactionDT, and the first few other
    numeric signals. String columns (ProductCD, card4/card6 raw, emails)
    are filtered out -- they should be handled upstream by the preprocessor
    or treated as categorical indices, not cast to float here.
    """
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    v_cols = [c for c in numeric_cols if c.startswith("V") and c[1:].isdigit()]
    # Rank V columns by target correlation (or variance if target missing).
    if v_cols and "isFraud" in df.columns and df["isFraud"].nunique() > 1:
        ranks = df[v_cols].corrwith(df["isFraud"]).abs().fillna(0.0)
    elif v_cols:
        ranks = df[v_cols].var().fillna(0.0)
    else:
        ranks = pd.Series(dtype=float)
    top_v = ranks.sort_values(ascending=False).index.tolist()[:47]
    # Pad with the most promising extras (all numeric).
    reserved = set(top_v)
    extras: list[str] = []
    for c in (
        "TransactionAmt__log1p",
        "TransactionAmt",
        "TransactionDT",
        "C1",
        "C13",
        "C14",
        "D1",
        "D15",
    ):
        if c in df.columns and c not in reserved and c in numeric_cols:
            extras.append(c)
            reserved.add(c)
        if len(top_v) + len(extras) >= 50:
            break
    cols = (top_v + extras)[:50]
    return cols


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class HeteroGraphBuilder:
    """Construct a :class:`HeteroData` graph from a processed IEEE-CIS frame.

    Parameters
    ----------
    config
        Optional :class:`AppConfig`. Defaults to :func:`load_config`.
    n_merchant_clusters
        Number of K-means clusters to use for the synthetic merchant nodes.
    n_ip_bins
        Per-axis quantile bin count for id_01 / id_02 -> IP bucket.
    cap_shared_edges_per_entity
        Maximum number of pairwise card-card edges to emit from a single
        shared address or device. Protects against quadratic blow-up on
        very popular entities (e.g. ProxyNet IP ranges).
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        n_merchant_clusters: int = 20,
        n_ip_bins: int = 10,
        cap_shared_edges_per_entity: int = 200,
    ) -> None:
        self.config = config or load_config()
        self.n_merchant_clusters = n_merchant_clusters
        self.n_ip_bins = n_ip_bins
        self.cap_shared_edges_per_entity = cap_shared_edges_per_entity
        self.state = GraphBuilderState()
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_hetero_data(
        self,
        df: pd.DataFrame,
        *,
        train_mask: pd.Series | None = None,
        val_mask: pd.Series | None = None,
        test_mask: pd.Series | None = None,
    ) -> HeteroData:
        """End-to-end build: indices -> features -> edges -> HeteroData.

        Parameters
        ----------
        df
            A processed, NaN-free frame (output of ``IEEECISPreprocessor``).
        train_mask, val_mask, test_mask
            Optional boolean row masks aligned with ``df``. Attached to
            the returned ``data['transaction']`` as masks.
        """
        log.info("graph_build_start", rows=len(df), cols=df.shape[1])

        indices = self.build_node_indices(df)
        features = self.build_node_features(df, indices)
        edges = self.build_edge_indices(df, indices)

        data = HeteroData()
        for node_type in NODE_TYPES:
            data[node_type].x = features[node_type]
            data[node_type].num_nodes = features[node_type].shape[0]

        for edge_type, edge_index in edges.items():
            data[edge_type].edge_index = edge_index
            data[edge_type].num_edges = int(edge_index.shape[1])

        # Labels + masks on transaction nodes (ordered by TransactionID).
        tx_id_to_idx = indices["transaction"]
        order = [k for k, _ in sorted(tx_id_to_idx.items(), key=lambda kv: kv[1])]
        df_ordered = df.set_index("TransactionID").loc[order]
        if self.config.dataset.target in df_ordered.columns:
            data["transaction"].y = torch.as_tensor(
                df_ordered[self.config.dataset.target].to_numpy(), dtype=torch.int8
            )

        for mask_name, mask in (
            ("train_mask", train_mask),
            ("val_mask", val_mask),
            ("test_mask", test_mask),
        ):
            if mask is not None:
                # Re-index onto the TransactionID-sorted order.
                m = mask.copy()
                m.index = df["TransactionID"].to_numpy()
                m = m.loc[order].astype(bool).to_numpy()
                data["transaction"][mask_name] = torch.as_tensor(m, dtype=torch.bool)

        log.info(
            "graph_build_complete",
            **{f"n_{nt}": int(data[nt].num_nodes) for nt in NODE_TYPES},
            **{f"e_{et[1]}": int(data[et].num_edges) for et in EDGE_TYPES},
        )
        self._fitted = True
        return data

    # ------------------------------------------------------------------
    # Stage 1: node indices
    # ------------------------------------------------------------------

    def build_node_indices(self, df: pd.DataFrame) -> dict[str, dict[Any, int]]:
        """Assign integer node indices for each node type.

        Returns
        -------
        dict[str, dict[Any, int]]
            For each node type, a mapping from entity key -> node index.
        """
        indices: dict[str, dict[Any, int]] = {}

        # transaction: one node per TransactionID, in arrival order.
        tx_sorted = df.sort_values(self.config.dataset.time_column, kind="mergesort")
        indices["transaction"] = {
            tid: i for i, tid in enumerate(tx_sorted["TransactionID"].tolist())
        }

        # card: one node per unique card1.
        indices["card"] = self._build_unique_index(df, "card1")

        # address: one node per unique (addr1, addr2) tuple.
        addr_key = list(
            zip(
                df["addr1"].fillna(-1).to_numpy(),
                df["addr2"].fillna(-1).to_numpy(),
                strict=True,
            )
        )
        indices["address"] = {k: i for i, k in enumerate(dict.fromkeys(addr_key))}

        # email: one node per unique P_emaildomain.
        indices["email"] = self._build_unique_index(df, "P_emaildomain")

        # device: one node per unique DeviceInfo (fall back to DeviceType if missing).
        dev_key = self._device_key_series(df)
        indices["device"] = {k: i for i, k in enumerate(dict.fromkeys(dev_key.tolist()))}

        # ip_address: bucket id_01 and id_02 (if fitted, use saved edges).
        ip_key = self._ip_key_series(df, fit=True)
        indices["ip_address"] = {k: i for i, k in enumerate(dict.fromkeys(ip_key.tolist()))}

        # merchant: K-means cluster over V307-V316 (fallback to any 10 V-cols).
        merchant_labels = self._fit_or_apply_merchant(df, fit=True)
        indices["merchant"] = {int(k): i for i, k in enumerate(np.unique(merchant_labels))}

        self.state.node_index_maps = indices
        log.info(
            "node_indices_built",
            **{f"n_{k}": len(v) for k, v in indices.items()},
        )
        return indices

    @staticmethod
    def _build_unique_index(df: pd.DataFrame, col: str) -> dict[Any, int]:
        values = df[col].astype("string").fillna("unknown").tolist()
        unique_vals = list(dict.fromkeys(values))  # preserve first-seen order
        return {v: i for i, v in enumerate(unique_vals)}

    @staticmethod
    def _device_key_series(df: pd.DataFrame) -> pd.Series:
        info = df.get("DeviceInfo")
        type_ = df.get("DeviceType")
        if info is None and type_ is None:
            return pd.Series(["unknown"] * len(df), index=df.index)
        info = info.astype("string").fillna("") if info is not None else ""
        type_ = type_.astype("string").fillna("") if type_ is not None else ""
        combined = (type_.astype(str) + "|" + info.astype(str)).replace("|", "unknown")
        return combined

    def _ip_key_series(self, df: pd.DataFrame, *, fit: bool) -> pd.Series:
        id01 = df.get("id_01") if "id_01" in df.columns else pd.Series(0, index=df.index)
        id02 = df.get("id_02") if "id_02" in df.columns else pd.Series(0, index=df.index)
        id01 = pd.to_numeric(id01, errors="coerce")
        id02 = pd.to_numeric(id02, errors="coerce")
        if fit:
            self.state.id01_bins = _quantile_bin_edges(id01, self.n_ip_bins)
            self.state.id02_bins = _quantile_bin_edges(id02, self.n_ip_bins)
        b1 = _apply_bin(id01, self.state.id01_bins)
        b2 = _apply_bin(id02, self.state.id02_bins)
        # Combine into a single string key for node indexing.
        return pd.Series([f"{a}|{b}" for a, b in zip(b1, b2, strict=True)], index=df.index)

    def _fit_or_apply_merchant(self, df: pd.DataFrame, *, fit: bool) -> np.ndarray:
        # Prefer V307-V316 (per plan), fall back to the first 10 V-cols if absent.
        preferred = [f"V{i}" for i in range(307, 317)]
        available = [c for c in preferred if c in df.columns]
        if len(available) < 2:
            available = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()][:10]
        if len(available) < 2:
            # Degenerate fallback: everyone in the same cluster.
            return np.zeros(len(df), dtype=np.int32)
        X = df[available].fillna(0).to_numpy(dtype=np.float32)
        if fit:
            n_clusters = min(self.n_merchant_clusters, max(2, len(df) // 1000))
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=self.config.project.seed,
                batch_size=1024,
                n_init=3,
            )
            kmeans.fit(X)
            self.state.merchant_kmeans = kmeans
        if self.state.merchant_kmeans is None:  # pragma: no cover -- defensive
            return np.zeros(len(df), dtype=np.int32)
        return self.state.merchant_kmeans.predict(X).astype(np.int32)

    # ------------------------------------------------------------------
    # Stage 2: node features
    # ------------------------------------------------------------------

    def build_node_features(
        self,
        df: pd.DataFrame,
        indices: dict[str, dict[Any, int]],
    ) -> dict[str, torch.Tensor]:
        """Emit per-node float32 feature tensors."""
        out: dict[str, torch.Tensor] = {}

        out["transaction"] = self._transaction_features(df, indices["transaction"])
        out["card"] = self._card_features(df, indices["card"])
        out["address"] = self._address_features(df, indices["address"])
        out["email"] = self._email_features(df, indices["email"])
        out["device"] = self._device_features(df, indices["device"])
        out["ip_address"] = self._ip_features(df, indices["ip_address"])
        out["merchant"] = self._merchant_features(indices["merchant"])

        # Sanity-check dims match the schema.
        for nt, dim in NODE_FEATURE_DIMS.items():
            actual = out[nt].shape[1]
            if actual != dim:
                log.warning("node_feature_dim_mismatch", node=nt, expected=dim, got=actual)
        return out

    def _transaction_features(self, df: pd.DataFrame, idx: dict[Any, int]) -> torch.Tensor:
        feature_cols = _select_transaction_feature_cols(df)
        n = len(idx)
        x = np.zeros((n, NODE_FEATURE_DIMS["transaction"]), dtype=np.float32)
        # Order rows by node index.
        ordered = df.set_index("TransactionID").reindex(
            [k for k, _ in sorted(idx.items(), key=lambda kv: kv[1])]
        )
        vals = ordered[feature_cols].fillna(0).to_numpy(dtype=np.float32)
        x[:, : vals.shape[1]] = vals
        return torch.as_tensor(x)

    def _card_features(self, df: pd.DataFrame, idx: dict[Any, int]) -> torch.Tensor:
        n = len(idx)
        x = np.zeros((n, NODE_FEATURE_DIMS["card"]), dtype=np.float32)
        # Take numeric mode per card1 for card2-card6 (ints), plus txn count + log1p avg amount.
        card_cols = [c for c in ("card2", "card3", "card4", "card5", "card6") if c in df.columns]
        grouped = df.groupby("card1", sort=False)
        for card1_val, node_idx in idx.items():
            if card1_val not in grouped.groups:
                continue
            sub = grouped.get_group(card1_val)
            for i, c in enumerate(card_cols[:5]):
                vals = pd.to_numeric(sub[c], errors="coerce").dropna()
                x[node_idx, i] = float(vals.mean()) if len(vals) else 0.0
            x[node_idx, 5] = float(len(sub))  # txn_count
            if "TransactionAmt" in sub.columns:
                x[node_idx, 6] = float(np.log1p(sub["TransactionAmt"].fillna(0).mean()))
            x[node_idx, 7] = float(
                sub[self.config.dataset.target].mean()
                if self.config.dataset.target in sub.columns
                else 0.0
            )
        return torch.as_tensor(x)

    def _address_features(self, df: pd.DataFrame, idx: dict[Any, int]) -> torch.Tensor:
        n = len(idx)
        x = np.zeros((n, NODE_FEATURE_DIMS["address"]), dtype=np.float32)
        addr_key = list(
            zip(
                df["addr1"].fillna(-1).to_numpy(),
                df["addr2"].fillna(-1).to_numpy(),
                strict=True,
            )
        )
        addr_df = df.assign(_addr_key=addr_key).groupby("_addr_key", sort=False)
        for key, node_idx in idx.items():
            if key not in addr_df.groups:
                continue
            sub = addr_df.get_group(key)
            x[node_idx, 0] = float(key[0]) if key[0] != -1 else 0.0
            x[node_idx, 1] = float(key[1]) if key[1] != -1 else 0.0
            for i, c in enumerate(("dist1", "dist2")):
                if c in sub.columns:
                    x[node_idx, 2 + i] = float(
                        pd.to_numeric(sub[c], errors="coerce").fillna(0).mean()
                    )
        return torch.as_tensor(x)

    def _email_features(self, df: pd.DataFrame, idx: dict[Any, int]) -> torch.Tensor:
        n = len(idx)
        x = np.zeros((n, NODE_FEATURE_DIMS["email"]), dtype=np.float32)
        counts = df["P_emaildomain"].astype("string").fillna("unknown").value_counts()
        max_count = max(int(counts.max()), 1)
        for domain, node_idx in idx.items():
            # popularity (normalized by max), is_free_provider, tld_risk
            x[node_idx, 0] = int(counts.get(domain, 0)) / max_count
            x[node_idx, 1] = 1.0 if str(domain).lower() in _FREE_EMAIL_DOMAINS else 0.0
            tld = str(domain).rsplit(".", maxsplit=1)[-1].lower()
            # Simple heuristic: cheap TLDs flagged higher-risk.
            x[node_idx, 2] = 1.0 if tld in {"ru", "cn", "xyz", "top", "click"} else 0.0
        return torch.as_tensor(x)

    def _device_features(self, df: pd.DataFrame, idx: dict[Any, int]) -> torch.Tensor:
        n = len(idx)
        x = np.zeros((n, NODE_FEATURE_DIMS["device"]), dtype=np.float32)
        dev_key = self._device_key_series(df)
        for key, node_idx in idx.items():
            mask = dev_key == key
            sub = df[mask]
            if sub.empty:
                continue
            x[node_idx, 0] = float(len(sub))  # frequency
            info_str = str(key).lower()
            x[node_idx, 1] = 1.0 if "windows" in info_str else 0.0
            x[node_idx, 2] = 1.0 if ("iphone" in info_str or "ipad" in info_str) else 0.0
            x[node_idx, 3] = 1.0 if "android" in info_str else 0.0
        return torch.as_tensor(x)

    def _ip_features(self, df: pd.DataFrame, idx: dict[Any, int]) -> torch.Tensor:
        n = len(idx)
        x = np.zeros((n, NODE_FEATURE_DIMS["ip_address"]), dtype=np.float32)
        ip_key = self._ip_key_series(df, fit=False)
        id_cols = [f"id_{i:02d}" for i in range(1, 7) if f"id_{i:02d}" in df.columns]
        for key, node_idx in idx.items():
            sub = df[ip_key == key]
            if sub.empty:
                continue
            for i, c in enumerate(id_cols[:6]):
                vals = pd.to_numeric(sub[c], errors="coerce").dropna()
                if len(vals):
                    x[node_idx, i] = float(vals.mean())
        return torch.as_tensor(x)

    def _merchant_features(self, idx: dict[int, int]) -> torch.Tensor:
        n = len(idx)
        x = np.zeros((n, NODE_FEATURE_DIMS["merchant"]), dtype=np.float32)
        if self.state.merchant_kmeans is not None:
            centers = self.state.merchant_kmeans.cluster_centers_.astype(np.float32)
            for cluster_id, node_idx in idx.items():
                center = centers[int(cluster_id)]
                x[node_idx, : min(10, center.size)] = center[:10]
        return torch.as_tensor(x)

    # ------------------------------------------------------------------
    # Stage 3: edge indices
    # ------------------------------------------------------------------

    def build_edge_indices(
        self,
        df: pd.DataFrame,
        indices: dict[str, dict[Any, int]],
    ) -> dict[tuple[str, str, str], torch.Tensor]:
        """Build one [2, E] edge_index tensor per edge type."""
        out: dict[tuple[str, str, str], torch.Tensor] = {}

        tx_ids = df["TransactionID"].tolist()
        tx_nodes = np.array([indices["transaction"][t] for t in tx_ids], dtype=np.int64)

        # ---- transaction -> entity edges ------------------------------
        out["transaction", "uses_card", "card"] = self._tx_entity_edges(
            tx_nodes,
            [indices["card"][v] for v in df["card1"].astype("string").fillna("unknown").tolist()],
        )

        addr_key = list(
            zip(
                df["addr1"].fillna(-1).to_numpy(),
                df["addr2"].fillna(-1).to_numpy(),
                strict=True,
            )
        )
        out["transaction", "from_address", "address"] = self._tx_entity_edges(
            tx_nodes, [indices["address"][k] for k in addr_key]
        )

        out["transaction", "from_email", "email"] = self._tx_entity_edges(
            tx_nodes,
            [
                indices["email"][v]
                for v in df["P_emaildomain"].astype("string").fillna("unknown").tolist()
            ],
        )

        dev_key = self._device_key_series(df).tolist()
        out["transaction", "from_device", "device"] = self._tx_entity_edges(
            tx_nodes, [indices["device"][k] for k in dev_key]
        )

        ip_key = self._ip_key_series(df, fit=False).tolist()
        out["transaction", "from_ip", "ip_address"] = self._tx_entity_edges(
            tx_nodes, [indices["ip_address"][k] for k in ip_key]
        )

        merchant_labels = self._fit_or_apply_merchant(df, fit=False)
        out["transaction", "at_merchant", "merchant"] = self._tx_entity_edges(
            tx_nodes, [indices["merchant"][int(lbl)] for lbl in merchant_labels]
        )

        # ---- card <-> card entity edges -------------------------------
        out["card", "shared_address", "card"] = self._pairwise_card_edges_via(
            df, indices["card"], key_col=None, key_series=addr_key
        )
        out["card", "shared_device", "card"] = self._pairwise_card_edges_via(
            df, indices["card"], key_col=None, key_series=dev_key
        )

        return out

    @staticmethod
    def _tx_entity_edges(tx_nodes: np.ndarray, entity_nodes: list[int]) -> torch.Tensor:
        assert len(tx_nodes) == len(entity_nodes), "row count mismatch"
        arr = np.stack([tx_nodes, np.asarray(entity_nodes, dtype=np.int64)], axis=0)
        return torch.as_tensor(arr, dtype=torch.long)

    def _pairwise_card_edges_via(
        self,
        df: pd.DataFrame,
        card_idx: dict[Any, int],
        *,
        key_col: str | None,
        key_series: list,
    ) -> torch.Tensor:
        """Emit card-card edges from cards that share a given entity key.

        We cap at :attr:`cap_shared_edges_per_entity` pairs per entity to
        avoid quadratic blow-up on popular hubs (e.g. a shared IP/device
        spanning thousands of cards).
        """
        cap = self.cap_shared_edges_per_entity
        src_rows: list[int] = []
        dst_rows: list[int] = []
        card_series = df["card1"].astype("string").fillna("unknown").tolist()
        # Group card nodes by the sharing key.
        by_key: dict[Any, list[int]] = {}
        for key, card_val in zip(key_series, card_series, strict=True):
            node_idx = card_idx[card_val]
            by_key.setdefault(key, []).append(node_idx)

        rng = np.random.default_rng(self.config.project.seed)
        for _key, nodes in by_key.items():
            unique_nodes = list(dict.fromkeys(nodes))
            if len(unique_nodes) < 2:
                continue
            # All unordered pairs; cap random sample if too many.
            n = len(unique_nodes)
            all_pairs_count = n * (n - 1) // 2
            if all_pairs_count <= cap:
                pairs = [
                    (unique_nodes[i], unique_nodes[j]) for i in range(n) for j in range(i + 1, n)
                ]
            else:
                sampled = rng.choice(unique_nodes, size=(cap, 2), replace=True)
                pairs = [(int(a), int(b)) for a, b in sampled if a != b]
            for a, b in pairs:
                src_rows.append(a)
                dst_rows.append(b)

        if not src_rows:
            return torch.empty((2, 0), dtype=torch.long)

        # Also emit the reverse direction so message passing works both ways.
        src_arr = np.asarray(src_rows + dst_rows, dtype=np.int64)
        dst_arr = np.asarray(dst_rows + src_rows, dtype=np.int64)
        return torch.as_tensor(np.stack([src_arr, dst_arr], axis=0))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: str | Path) -> None:
        self.state.to_file(path)
        log.info("graph_state_saved", path=str(path))

    @classmethod
    def load_state(cls, path: str | Path, config: AppConfig | None = None) -> HeteroGraphBuilder:
        gb = cls(config)
        gb.state = GraphBuilderState.from_file(path)
        gb._fitted = True
        return gb


__all__ = [
    "EDGE_TYPES",
    "NODE_FEATURE_DIMS",
    "NODE_TYPES",
    "GraphBuilderState",
    "HeteroGraphBuilder",
]
