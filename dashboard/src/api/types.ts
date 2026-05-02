/**
 * TypeScript schemas mirroring `fraud_detection.serving.schemas` (Python)
 * and `fraud_detection.agent.state` so the dashboard stays in sync with
 * the FastAPI app's contract.
 */

export type RiskLevel = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
export type AlertRiskLevel = Exclude<RiskLevel, "LOW">;
export type RecommendedAction = "approve" | "review" | "decline" | "escalate";
export type InvestigationDepth = "quick" | "standard" | "deep";

export interface FeatureContribution {
  feature: string;
  value: number;
  contribution: number;
}

export interface FraudPrediction {
  transaction_id: number | string;
  fraud_probability: number;
  fraud_score: number;
  risk_level: RiskLevel;
  is_fraud_predicted: boolean;
  threshold: number;
  top_features: FeatureContribution[];
  latency_ms: Record<string, number>;
  model_version: string;
  served_at?: string;
}

export interface FraudAlert {
  transaction_id: number | string;
  fraud_score: number;
  risk_level: AlertRiskLevel;
  transaction_amt: number;
  card_id?: string | number | null;
  timestamp?: string;
  top_features?: FeatureContribution[];
}

export interface HealthStatus {
  status: "ok" | "degraded" | "down";
  model_loaded: boolean;
  redis_connected: boolean;
  kafka_connected: boolean;
  ray_serve_active: boolean;
  uptime_seconds: number;
  version: string;
}

export interface ModelInfo {
  model_version: string;
  n_parameters: number;
  embedding_dim: number;
  n_features: number;
  feature_columns: string[];
  edge_types: string[];
  node_types: string[];
  train_metrics: Record<string, number>;
  feature_importance_top_k: Record<string, number>;
}

export interface TransactionRequest {
  transaction_id: number | string;
  transaction_dt: number;
  transaction_amt: number;
  card1?: number | null;
  product_cd?: string;
  P_emaildomain?: string | null;
  DeviceType?: string | null;
  [key: string]: unknown;
}

export interface EntityRisk {
  entity_type: "card" | "device" | "email" | "ip" | "merchant";
  entity_id: string;
  risk_score: number;
  contributing_factors: string[];
}

export interface FraudPattern {
  name:
    | "card_testing"
    | "account_takeover"
    | "collusion_ring"
    | "velocity_spike"
    | "geo_anomaly"
    | "none";
  confidence: number;
  rationale: string;
}

export interface SimilarCase {
  case_id: string;
  similarity: number;
  pattern: string;
  summary: string;
}

export interface ToolCallEntry {
  name: string;
  status: string;
  elapsed_ms: number;
}

export interface InvestigationReport {
  alert_id: string;
  transaction_id: number | string;
  risk_level: RiskLevel;
  depth: InvestigationDepth;
  fraud_score: number;

  summary: string;
  narrative: string;
  recommended_action: RecommendedAction;
  confidence: number;
  requires_human_review: boolean;

  entity_risks: EntityRisk[];
  matched_patterns: FraudPattern[];
  similar_cases: SimilarCase[];

  tools_used: string[];
  tool_calls: ToolCallEntry[];
  elapsed_ms: number;
  generated_at?: string;
  model: string;
}

/** Phase 6: optional history endpoint that the API exposes for first-load. */
export interface RecentResponse {
  predictions: FraudPrediction[];
  alerts: FraudAlert[];
  generated_at: string;
}
