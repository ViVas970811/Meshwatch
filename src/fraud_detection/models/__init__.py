"""Model architectures (Phase 3).

Public surface:

* :class:`FraudHeteroGNN` -- heterogeneous GNN with extractable 64-d
  embeddings (per-transaction).
* :class:`HeteroGNNLayer` -- one block of SAGEConv (tx-entity) + GATConv
  (entity-entity) with residual / LayerNorm / ELU / Dropout.
* :class:`FocalLoss` -- alpha=0.75, gamma=2.0 by default for IEEE-CIS.
* :class:`XGBoostFraudModel` -- sklearn-style wrapper around xgboost.
* :class:`FraudEnsemble` -- two-stage GNN+XGBoost ensemble.
"""

from fraud_detection.models.ensemble import EnsembleArtifacts, FraudEnsemble
from fraud_detection.models.gnn_layers import HeteroGNNLayer
from fraud_detection.models.hetero_gnn import FraudHeteroGNN
from fraud_detection.models.losses import FocalLoss
from fraud_detection.models.xgboost_model import XGBoostConfig, XGBoostFraudModel

__all__ = [
    "EnsembleArtifacts",
    "FocalLoss",
    "FraudEnsemble",
    "FraudHeteroGNN",
    "HeteroGNNLayer",
    "XGBoostConfig",
    "XGBoostFraudModel",
]
