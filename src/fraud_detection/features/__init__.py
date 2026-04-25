"""Feature engineering (Phase 2).

Public surface:

* :class:`TemporalFeatureBuilder` -- temporal + amount features (39).
* :class:`AggregatedFeatureBuilder` -- per-entity aggregates + identity-risk
  flags (52).
* :class:`GraphFeatureBuilder` -- structural features computed on the
  card-card projection (28).
* :class:`FeaturePipeline` -- orchestrator that fits + transforms all three
  builders and emits the 119 engineered columns the GNN/XGBoost pipeline
  expects.
"""

from fraud_detection.features.aggregated import (
    AGGREGATED_ADDR_FEATURES,
    AGGREGATED_CARD_FEATURES,
    AGGREGATED_DEVICE_FEATURES,
    AGGREGATED_EMAIL_FEATURES,
    AGGREGATED_FEATURES,
    ALL_AGGREGATED_FEATURES,
    C_FAMILY_FEATURES,
    IDENTITY_FEATURES,
    AggregatedFeatureBuilder,
    AggregatedState,
)
from fraud_detection.features.graph_features import (
    GRAPH_FEATURES,
    GraphFeatureBuilder,
    GraphFeatureState,
)
from fraud_detection.features.pipeline import (
    ALL_ENGINEERED_FEATURES,
    FeaturePipeline,
    FeaturePipelineState,
)
from fraud_detection.features.temporal import (
    ALL_TEMPORAL_AMOUNT_FEATURES,
    AMOUNT_FEATURES,
    TEMPORAL_FEATURES,
    TemporalFeatureBuilder,
    TemporalState,
)

__all__ = [
    "AGGREGATED_ADDR_FEATURES",
    "AGGREGATED_CARD_FEATURES",
    "AGGREGATED_DEVICE_FEATURES",
    "AGGREGATED_EMAIL_FEATURES",
    "AGGREGATED_FEATURES",
    "ALL_AGGREGATED_FEATURES",
    "ALL_ENGINEERED_FEATURES",
    "ALL_TEMPORAL_AMOUNT_FEATURES",
    "AMOUNT_FEATURES",
    "C_FAMILY_FEATURES",
    "GRAPH_FEATURES",
    "IDENTITY_FEATURES",
    "TEMPORAL_FEATURES",
    "AggregatedFeatureBuilder",
    "AggregatedState",
    "FeaturePipeline",
    "FeaturePipelineState",
    "GraphFeatureBuilder",
    "GraphFeatureState",
    "TemporalFeatureBuilder",
    "TemporalState",
]
