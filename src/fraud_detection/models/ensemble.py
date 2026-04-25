"""GNN + XGBoost two-stage ensemble.

Per the implementation plan (page 7) the ensemble takes::

    [GNN_embedding(64) + tabular_features(119) + V_features(~140)] = ~323

and feeds them into XGBoost (see :mod:`xgboost_model`). The expected lift
over GNN alone is +5 AUPRC points (0.65 -> 0.70).

This module wires the two stages together with persistence helpers so that
the entire serving graph (preprocessor -> graph builder -> feature pipeline
-> GNN -> XGBoost) can be pickled to a single artifact bundle.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import HeteroData

from fraud_detection.models.hetero_gnn import FraudHeteroGNN
from fraud_detection.models.xgboost_model import XGBoostFraudModel
from fraud_detection.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class EnsembleArtifacts:
    """Serialisable bundle for the ensemble (everything except the GNN weights).

    The GNN weights themselves live in a separate ``state_dict.pt`` file on
    disk -- pickling a ``torch.nn.Module`` directly is fragile across
    PyTorch versions.
    """

    gnn_config: dict[str, Any] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=list)
    gnn_embedding_columns: list[str] = field(default_factory=list)


class FraudEnsemble:
    """Combines a trained :class:`FraudHeteroGNN` with an XGBoost head.

    Workflow:

    1. ``fit_xgboost`` -- given a trained GNN, extract embeddings on the
       training graph + concatenate with tabular features, then fit XGBoost.
    2. ``predict_proba`` -- score new transactions: GNN forward -> embed ->
       concat with tabular -> XGBoost predict_proba.

    Persisted via :meth:`save` (writes ``ensemble/`` directory with
    ``gnn.pt``, ``xgb.pkl``, ``artifacts.pkl``).
    """

    def __init__(
        self,
        gnn: FraudHeteroGNN,
        xgboost_model: XGBoostFraudModel | None = None,
        *,
        embedding_prefix: str = "gnn_emb",
    ) -> None:
        self.gnn = gnn
        self.xgb = xgboost_model or XGBoostFraudModel()
        self.embedding_prefix = embedding_prefix
        self.feature_columns: list[str] = []
        self.gnn_embedding_columns: list[str] = []

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def gnn_embeddings(
        self,
        data: HeteroData,
        *,
        target_indices: torch.Tensor | None = None,
        device: str | torch.device = "cpu",
    ) -> np.ndarray:
        """Return ``(N, embedding_dim)`` embeddings as a NumPy array."""
        self.gnn.eval()
        self.gnn.to(device)
        data = data.to(device)
        emb = self.gnn.get_embeddings(data, target_indices=target_indices)
        return emb.cpu().numpy()

    # ------------------------------------------------------------------
    # Train / predict
    # ------------------------------------------------------------------

    def _stack(
        self,
        embeddings: np.ndarray,
        tabular: np.ndarray,
    ) -> np.ndarray:
        return np.concatenate([embeddings.astype(np.float32), tabular.astype(np.float32)], axis=1)

    def _ensure_columns(self, tabular_columns: Iterable[str], embedding_dim: int) -> list[str]:
        emb_cols = [f"{self.embedding_prefix}_{i:03d}" for i in range(embedding_dim)]
        self.gnn_embedding_columns = emb_cols
        self.feature_columns = emb_cols + list(tabular_columns)
        return self.feature_columns

    def fit_xgboost(
        self,
        *,
        train_data: HeteroData,
        train_indices: torch.Tensor,
        train_tabular: np.ndarray,
        train_y: np.ndarray,
        tabular_columns: Iterable[str],
        val_data: HeteroData | None = None,
        val_indices: torch.Tensor | None = None,
        val_tabular: np.ndarray | None = None,
        val_y: np.ndarray | None = None,
        device: str | torch.device = "cpu",
    ) -> FraudEnsemble:
        """Extract GNN embeddings, concat with tabular, fit XGBoost.

        Both train and val arrays must already be in the same row order as
        ``train_indices`` / ``val_indices`` -- the embedding extraction
        respects that ordering.
        """
        train_emb = self.gnn_embeddings(train_data, target_indices=train_indices, device=device)
        log.info(
            "gnn_train_embeddings_extracted",
            shape=tuple(train_emb.shape),
        )
        cols = self._ensure_columns(tabular_columns, embedding_dim=train_emb.shape[1])
        X_train = self._stack(train_emb, train_tabular)

        eval_X = eval_y = None
        if val_data is not None and val_indices is not None and val_tabular is not None:
            val_emb = self.gnn_embeddings(val_data, target_indices=val_indices, device=device)
            eval_X = self._stack(val_emb, val_tabular)
            eval_y = val_y

        self.xgb.fit(
            X_train,
            train_y,
            X_val=eval_X,
            y_val=eval_y,
            feature_names=cols,
        )
        return self

    @torch.no_grad()
    def predict_proba(
        self,
        data: HeteroData,
        tabular: np.ndarray,
        *,
        target_indices: torch.Tensor | None = None,
        device: str | torch.device = "cpu",
    ) -> np.ndarray:
        emb = self.gnn_embeddings(data, target_indices=target_indices, device=device)
        X = self._stack(emb, tabular)
        return self.xgb.predict_proba(X)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, dir_path: str | Path, *, gnn_init_kwargs: dict[str, Any]) -> None:
        """Persist the ensemble.

        Writes:

        * ``gnn.pt``           -- ``state_dict`` of the GNN
        * ``xgb.pkl``          -- :class:`XGBoostFraudModel` pickle
        * ``artifacts.pkl``    -- feature column names + GNN init kwargs
        """
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.gnn.state_dict(), dir_path / "gnn.pt")
        self.xgb.save(dir_path / "xgb.pkl")
        artifacts = EnsembleArtifacts(
            gnn_config=gnn_init_kwargs,
            feature_columns=self.feature_columns,
            gnn_embedding_columns=self.gnn_embedding_columns,
        )
        with (dir_path / "artifacts.pkl").open("wb") as f:
            pickle.dump(artifacts, f)
        log.info("ensemble_saved", dir=str(dir_path))

    @classmethod
    def load(cls, dir_path: str | Path) -> FraudEnsemble:
        dir_path = Path(dir_path)
        with (dir_path / "artifacts.pkl").open("rb") as f:
            artifacts: EnsembleArtifacts = pickle.load(f)
        gnn = FraudHeteroGNN(**artifacts.gnn_config)
        gnn.load_state_dict(torch.load(dir_path / "gnn.pt", weights_only=True))
        xgb_model = XGBoostFraudModel.load(dir_path / "xgb.pkl")
        inst = cls(gnn, xgb_model)
        inst.feature_columns = artifacts.feature_columns
        inst.gnn_embedding_columns = artifacts.gnn_embedding_columns
        log.info("ensemble_loaded", dir=str(dir_path), n_features=len(inst.feature_columns))
        return inst


__all__ = ["EnsembleArtifacts", "FraudEnsemble"]
