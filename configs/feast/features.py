"""Feast feature views for the 119 engineered fraud-detection features.

Applied via::

    cd configs/feast
    feast apply

On apply, Feast registers:

* One Entity per ``TransactionID``.
* Three FeatureViews (temporal+amount / aggregated+identity / graph),
  each sourced from ``data/graphs/features.parquet``.

The file is consumed both offline (training joins) and online (serving
lookups via the SQLite online store).
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

# --- Source --------------------------------------------------------------
# Resolved relative to the feature store YAML at runtime; Feast joins the
# registry project dir with this path.
_FEATURES_PATH = Path("../../data/graphs/features.parquet")

features_source = FileSource(
    name="engineered_features",
    path=str(_FEATURES_PATH),
    timestamp_field="event_timestamp",
)

# --- Entity --------------------------------------------------------------
transaction = Entity(
    name="transaction",
    join_keys=["TransactionID"],
    description="A single IEEE-CIS transaction row.",
)


# --- Feature schemas (mirror the canonical lists from the builders) ------
# We redeclare the columns here so Feast's type checker can validate the
# parquet schema at ``feast apply`` time.


def _f32(names: list[str]) -> list[Field]:
    return [Field(name=n, dtype=Float32) for n in names]


TEMPORAL_COLUMNS = [
    "feat_hour_sin",
    "feat_hour_cos",
    "feat_dow_sin",
    "feat_dow_cos",
    "feat_hour",
    "feat_dow",
    "feat_is_weekend",
    "feat_is_night",
    "feat_is_business_hours",
    "feat_seconds_since_last_txn",
    "feat_seconds_since_last_card_txn",
    "feat_seconds_since_first_card_txn",
    "feat_txn_count_1h",
    "feat_txn_count_24h",
    "feat_txn_count_7d",
    "feat_card_txn_count_1h",
    "feat_card_txn_count_24h",
    "feat_velocity_1h",
    "feat_velocity_24h",
    "feat_velocity_7d",
    "feat_interarrival_mean_card",
    "feat_interarrival_std_card",
    "feat_interarrival_last_card",
    "feat_amt_log1p",
    "feat_amt_cents",
    "feat_amt_zscore",
    "feat_amt_percentile",
    "feat_amt_cum_1h",
    "feat_amt_cum_24h",
    "feat_amt_cum_7d",
    "feat_amt_card_cum",
    "feat_amt_vs_card_mean",
    "feat_amt_vs_card_max",
    "feat_amt_ratio_to_card_min",
    "feat_amt_ratio_to_card_max",
    "feat_amt_above_p95",
    "feat_amt_below_p05",
    "feat_amt_acceleration",
    "feat_amt_jerk",
]

AGGREGATED_COLUMNS = [
    "feat_card_mean_amt",
    "feat_card_median_amt",
    "feat_card_std_amt",
    "feat_card_max_amt",
    "feat_card_count",
    "feat_card_fraud_rate",
    "feat_card_unique_addr",
    "feat_card_unique_emails",
    "feat_card_unique_devices",
    "feat_card_unique_products",
    "feat_card_days_active",
    "feat_card_txn_freq",
    "feat_email_mean_amt",
    "feat_email_median_amt",
    "feat_email_std_amt",
    "feat_email_count",
    "feat_email_fraud_rate",
    "feat_email_unique_cards",
    "feat_email_unique_addrs",
    "feat_addr_mean_amt",
    "feat_addr_median_amt",
    "feat_addr_std_amt",
    "feat_addr_count",
    "feat_addr_fraud_rate",
    "feat_addr_n_cards",
    "feat_device_mean_amt",
    "feat_device_median_amt",
    "feat_device_std_amt",
    "feat_device_count",
    "feat_device_fraud_rate",
    "feat_device_n_cards",
    "feat_c1_mean_card",
    "feat_c13_mean_card",
    "feat_c14_mean_card",
    "feat_m_flag_mean_card",
    "feat_c_total_per_txn",
    "feat_id_01_bin_risk",
    "feat_id_02_bin_risk",
    "feat_device_is_windows",
    "feat_device_is_ios",
    "feat_device_is_android",
    "feat_device_is_mobile",
    "feat_is_proxy_heuristic",
    "feat_email_mismatch",
    "feat_high_risk_tld",
    "feat_suspicious_hour",
    "feat_identity_missing",
    "feat_new_device_for_card",
    "feat_multi_card_on_device",
    "feat_v_feature_zscore_max",
    "feat_v_feature_zscore_mean",
    "feat_amt_outlier_flag",
]

GRAPH_COLUMNS = [
    "feat_gr_tx_degree",
    "feat_gr_card_degree",
    "feat_gr_addr_degree",
    "feat_gr_email_degree",
    "feat_gr_device_degree",
    "feat_gr_ip_degree",
    "feat_gr_card_pagerank",
    "feat_gr_addr_pagerank",
    "feat_gr_email_pagerank",
    "feat_gr_device_pagerank",
    "feat_gr_ip_pagerank",
    "feat_gr_card_betweenness",
    "feat_gr_addr_betweenness",
    "feat_gr_email_betweenness",
    "feat_gr_device_betweenness",
    "feat_gr_card_closeness",
    "feat_gr_addr_closeness",
    "feat_gr_component_size",
    "feat_gr_nbr_fraud_card_1h",
    "feat_gr_nbr_fraud_addr_1h",
    "feat_gr_nbr_fraud_email_1h",
    "feat_gr_nbr_fraud_device_1h",
    "feat_gr_nbr_fraud_card_2h",
    "feat_gr_nbr_fraud_addr_2h",
    "feat_gr_ring_member",
    "feat_gr_ring_size",
    "feat_gr_avg_nbr_deg_card",
    "feat_gr_avg_nbr_deg_addr",
]


# --- Feature views -------------------------------------------------------
temporal_view = FeatureView(
    name="transaction_temporal",
    entities=[transaction],
    ttl=timedelta(days=7),
    schema=[Field(name="TransactionID", dtype=Int64)] + _f32(TEMPORAL_COLUMNS),
    online=True,
    source=features_source,
    tags={"family": "temporal_amount", "phase": "2"},
)

aggregated_view = FeatureView(
    name="transaction_aggregated",
    entities=[transaction],
    ttl=timedelta(days=7),
    schema=[Field(name="TransactionID", dtype=Int64)] + _f32(AGGREGATED_COLUMNS),
    online=True,
    source=features_source,
    tags={"family": "aggregated_identity", "phase": "2"},
)

graph_view = FeatureView(
    name="transaction_graph",
    entities=[transaction],
    ttl=timedelta(days=7),
    schema=[Field(name="TransactionID", dtype=Int64)] + _f32(GRAPH_COLUMNS),
    online=True,
    source=features_source,
    tags={"family": "graph_structural", "phase": "2"},
)
