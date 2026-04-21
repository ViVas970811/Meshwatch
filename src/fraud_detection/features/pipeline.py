"""Feature-engineering pipeline orchestrator.

Runs the three Phase 2 feature builders in order and concatenates their
outputs into a single frame of **119 engineered columns** alongside the
input identifiers::

    TransactionID, isFraud, TransactionDT, <feat_*>...

The pipeline is stateful: ``fit_transform`` on training data captures
entity statistics + graph topology; ``transform`` re-applies them to
unseen data without any target leakage.

The output DataFrame is what the downstream GNN and XGBoost models
consume. It is also the source of truth for Feast feature views (Phase 2
acceptance: "Feast feature store apply succeeds").
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fraud_detection.features.aggregated import (
    ALL_AGGREGATED_FEATURES,
    AggregatedFeatureBuilder,
)
from fraud_detection.features.graph_features import (
    GRAPH_FEATURES,
    GraphFeatureBuilder,
)
from fraud_detection.features.temporal import (
    ALL_TEMPORAL_AMOUNT_FEATURES,
    TemporalFeatureBuilder,
)
from fraud_detection.utils.logging import get_logger

log = get_logger(__name__)


ALL_ENGINEERED_FEATURES: tuple[str, ...] = (
    ALL_TEMPORAL_AMOUNT_FEATURES + ALL_AGGREGATED_FEATURES + GRAPH_FEATURES
)


@dataclass
class FeaturePipelineState:
    """Fitted state for all 3 builders, serialisable as one blob."""

    temporal: TemporalFeatureBuilder
    aggregated: AggregatedFeatureBuilder
    graph: GraphFeatureBuilder


class FeaturePipeline:
    """End-to-end orchestrator for the 119 engineered features."""

    def __init__(
        self,
        temporal: TemporalFeatureBuilder | None = None,
        aggregated: AggregatedFeatureBuilder | None = None,
        graph: GraphFeatureBuilder | None = None,
    ) -> None:
        self.temporal = temporal or TemporalFeatureBuilder()
        self.aggregated = aggregated or AggregatedFeatureBuilder()
        self.graph = graph or GraphFeatureBuilder()
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self, df: pd.DataFrame, *, training_mask: pd.Series | None = None
    ) -> pd.DataFrame:
        """Fit all 3 builders on ``df`` and return the engineered frame.

        Parameters
        ----------
        df
            Output of :class:`IEEECISPreprocessor.fit_transform` -- must
            have the target, time, and identifier columns present.
        training_mask
            Boolean series aligned to ``df``. When provided, target-encoded
            features (fraud rates) only use training rows. Recommended to
            avoid leakage into val / test splits.
        """
        log.info("feature_pipeline_fit_start", rows=len(df), cols=df.shape[1])
        temporal_out = self.temporal.fit_transform(df)
        aggregated_out = self.aggregated.fit_transform(df)
        graph_out = self.graph.fit_transform(df, training_mask=training_mask)
        self._fitted = True
        return self._concat(df, temporal_out, aggregated_out, graph_out)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            msg = "FeaturePipeline must be fit before transform."
            raise RuntimeError(msg)
        log.info("feature_pipeline_transform_start", rows=len(df))
        temporal_out = self.temporal.transform(df)
        aggregated_out = self.aggregated.transform(df)
        graph_out = self.graph.transform(df)
        return self._concat(df, temporal_out, aggregated_out, graph_out)

    def _concat(
        self,
        df: pd.DataFrame,
        temporal_out: pd.DataFrame,
        aggregated_out: pd.DataFrame,
        graph_out: pd.DataFrame,
    ) -> pd.DataFrame:
        # Keep identifiers so downstream consumers can join back.
        id_cols = [c for c in ("TransactionID", "isFraud", "TransactionDT") if c in df.columns]
        combined = pd.concat(
            [df[id_cols].reset_index(drop=True)]
            + [x.reset_index(drop=True) for x in (temporal_out, aggregated_out, graph_out)],
            axis=1,
        )
        log.info(
            "feature_pipeline_concat_done",
            total_cols=combined.shape[1],
            n_temporal=temporal_out.shape[1],
            n_aggregated=aggregated_out.shape[1],
            n_graph=graph_out.shape[1],
        )
        return combined

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        if not self._fitted:
            msg = "Cannot save a pipeline that has not been fit."
            raise RuntimeError(msg)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        log.info("feature_pipeline_saved", path=str(path))

    @classmethod
    def load(cls, path: str | Path) -> FeaturePipeline:
        with Path(path).open("rb") as f:
            pipeline: FeaturePipeline = pickle.load(f)
        return pipeline


__all__ = [
    "ALL_ENGINEERED_FEATURES",
    "FeaturePipeline",
    "FeaturePipelineState",
]
