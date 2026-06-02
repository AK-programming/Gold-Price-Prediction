/* ──────────────────────────────────────────────
   GoldSight AI — API Response Types
   ────────────────────────────────────────────── */

/** GET /api/v1/forecast */
export interface ForecastResponse {
  dates: string[];
  predictions: number[];
  confidence_low: number[];
  confidence_high: number[];
  current_price: number;
  model_used: string;
  generated_at: string;
}

/** Single data-point inside the historical response */
export interface HistoricalPoint {
  date: string;
  actual: number;
  predicted?: number;
}

/** GET /api/v1/historical */
export interface HistoricalResponse {
  data: HistoricalPoint[];
  count: number;
}

/** One model entry inside /api/v1/models */
export interface ModelMetrics {
  rmse: number;
  mae: number;
  mape: number;
  r2?: number;
  directional_accuracy?: number;
  [key: string]: number | undefined;
}

export interface ModelSummary {
  name: string;
  metrics: ModelMetrics;
  checkpoint_exists: boolean;
}

/** GET /api/v1/models */
export interface ModelsListResponse {
  models: ModelSummary[];
  latest_run: string;
}

/** Single feature entry from SHAP */
export interface ShapFeature {
  feature: string;
  importance: number;
}

/** GET /api/v1/explainability/shap */
export interface ShapResponse {
  feature_importance: ShapFeature[];
  num_samples: number;
  num_features: number;
}

/** GET /api/v1/explainability/attention */
export interface AttentionResponse {
  temporal_importance: number[];
  sequence_length: number;
  input_window: number;
}
