"use client";

import React, { useState, useEffect, useCallback, useMemo } from "react";
import { motion } from "framer-motion";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
  Legend,
} from "recharts";
import { fetchJson, postJson, waitForBackend } from "./api";
import type {
  DataStatusResponse,
  ForecastResponse,
  HistoricalResponse,
  HistoricalPoint,
  ModelsListResponse,
  ShapResponse,
  AttentionResponse,
} from "./types";

/* ═══════════════════════════════════════════════════════════════
   Mock Data — Used when the backend is unreachable
   ═══════════════════════════════════════════════════════════════ */

function generateMockHistorical(days: number): HistoricalPoint[] {
  const data: HistoricalPoint[] = [];
  const now = new Date();
  let price = 2320;
  for (let i = days; i >= 0; i--) {
    const d = new Date(now);
    d.setDate(d.getDate() - i);
    price += (Math.random() - 0.48) * 18;
    price = Math.max(2200, Math.min(2500, price));
    const predicted = i < 30 ? price + (Math.random() - 0.5) * 30 : undefined;
    data.push({
      date: d.toISOString().split("T")[0],
      actual: parseFloat(price.toFixed(2)),
      predicted: predicted ? parseFloat(predicted.toFixed(2)) : undefined,
    });
  }
  return data;
}

const generateHourlyData = (basePrice: number) => {
  const data = [];
  let a = basePrice + (Math.random() - 0.5) * 50;
  for (let i = 0; i < 60; i++) {
    a += (Math.random() - 0.5) * 10;
    const p = a + (Math.random() - 0.5) * 15;
    data.push({
      time: i,
      actual: i < 50 ? a : null,
      predicted: i >= 45 ? p : (i < 50 ? a + (Math.random() - 0.5) * 5 : null),
    });
  }
  return data;
};

const MOCK_HISTORICAL: HistoricalResponse = {
  data: generateMockHistorical(365),
  count: 366,
};

const MOCK_FORECAST: ForecastResponse = (() => {
  const dates: string[] = [];
  const predictions: number[] = [];
  const confLow: number[] = [];
  const confHigh: number[] = [];
  const now = new Date();
  let p = 2385;
  for (let i = 1; i <= 10; i++) {
    const d = new Date(now);
    d.setDate(d.getDate() + i);
    dates.push(d.toISOString().split("T")[0]);
    p += (Math.random() - 0.45) * 12;
    predictions.push(parseFloat(p.toFixed(2)));
    confLow.push(parseFloat((p - 20 - Math.random() * 15).toFixed(2)));
    confHigh.push(parseFloat((p + 20 + Math.random() * 15).toFixed(2)));
  }
  return {
    dates,
    predictions,
    confidence_low: confLow,
    confidence_high: confHigh,
    current_price: 2381.45,
    model_used: "Transformer-LSTM Ensemble",
    generated_at: new Date().toISOString(),
  };
})();

const MOCK_MODELS: ModelsListResponse = {
  models: [
    {
      name: "Transformer-LSTM Ensemble",
      metrics: { rmse: 12.34, mae: 9.21, mape: 0.39, r2: 0.967, directional_accuracy: 0.88 },
      checkpoint_exists: true,
    },
    {
      name: "LSTM (128 units)",
      metrics: { rmse: 15.87, mae: 11.63, mape: 0.49, r2: 0.952, directional_accuracy: 0.84 },
      checkpoint_exists: true,
    },
    {
      name: "GRU Attention",
      metrics: { rmse: 14.22, mae: 10.45, mape: 0.44, r2: 0.958, directional_accuracy: 0.86 },
      checkpoint_exists: true,
    },
    {
      name: "XGBoost Regressor",
      metrics: { rmse: 18.91, mae: 14.02, mape: 0.59, r2: 0.934, directional_accuracy: 0.81 },
      checkpoint_exists: false,
    },
    {
      name: "Linear Baseline",
      metrics: { rmse: 24.56, mae: 19.78, mape: 0.83, r2: 0.891, directional_accuracy: 0.73 },
      checkpoint_exists: false,
    },
  ],
  latest_run: new Date().toISOString(),
};

const MOCK_SHAP: ShapResponse = {
  feature_importance: [
    { feature: "Close_Lag1", importance: 0.32 },
    { feature: "MA_20", importance: 0.19 },
    { feature: "RSI_14", importance: 0.14 },
    { feature: "USD_Index", importance: 0.11 },
    { feature: "Volume", importance: 0.08 },
    { feature: "MACD", importance: 0.06 },
    { feature: "Bollinger_Upper", importance: 0.04 },
    { feature: "Treasury_10Y", importance: 0.03 },
    { feature: "SP500", importance: 0.02 },
    { feature: "VIX", importance: 0.01 },
  ],
  num_samples: 500,
  num_features: 10,
};

const MOCK_ATTENTION: AttentionResponse = {
  temporal_importance: [
    0.02, 0.03, 0.04, 0.04, 0.05, 0.06, 0.05, 0.07, 0.06, 0.08, 0.07, 0.09, 0.08, 0.1, 0.09,
    0.11, 0.12, 0.14, 0.16, 0.18, 0.21, 0.25, 0.3, 0.37, 0.42, 0.51, 0.6, 0.72, 0.85, 0.95,
  ],
  sequence_length: 30,
  input_window: 30,
};

/* ═══════════════════════════════════════════════════════════════
   Helpers
   ═══════════════════════════════════════════════════════════════ */

const fmt = (n: number) =>
  n.toLocaleString("en-US", { style: "currency", currency: "USD" });

const fmtCompact = (n: number) =>
  n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

const pctFmt = (n: number | undefined | null) =>
  typeof n === "number" && !Number.isNaN(n) ? `${n.toFixed(2)}%` : "—";

const forecastDates = (forecast?: ForecastResponse | null) =>
  Array.isArray(forecast?.dates) ? forecast.dates : MOCK_FORECAST.dates;
const forecastPredictions = (forecast?: ForecastResponse | null) =>
  Array.isArray(forecast?.predictions) ? forecast.predictions : MOCK_FORECAST.predictions;
const forecastConfidenceLow = (forecast?: ForecastResponse | null) =>
  Array.isArray(forecast?.confidence_low) ? forecast.confidence_low : MOCK_FORECAST.confidence_low;
const forecastConfidenceHigh = (forecast?: ForecastResponse | null) =>
  Array.isArray(forecast?.confidence_high) ? forecast.confidence_high : MOCK_FORECAST.confidence_high;
const safeNumber = (value: number | undefined | null, fallback = 0) =>
  typeof value === "number" && !Number.isNaN(value) ? value : fallback;
const formatNumber = (value: number | undefined | null, digits = 2) =>
  safeNumber(value).toFixed(digits);

type TimeRange = "30d" | "90d" | "180d" | "365d";

const RANGE_DAYS: Record<TimeRange, number> = {
  "30d": 30,
  "90d": 90,
  "180d": 180,
  "365d": 365,
};

/* framer-motion presets */
const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0 },
};

const stagger = {
  visible: { transition: { staggerChildren: 0.08 } },
};

/* ═══════════════════════════════════════════════════════════════
   Custom Recharts Tooltip
   ═══════════════════════════════════════════════════════════════ */

interface TooltipPayloadEntry {
  name?: string;
  value?: number;
  color?: string;
  dataKey?: string;
}

interface ChartTooltipProps {
  active?: boolean;
  payload?: TooltipPayloadEntry[];
  label?: string;
}

function ChartTooltip({ active, payload, label }: ChartTooltipProps) {
  if (!active || !payload?.length) return null;
  return (
    <div className="custom-tooltip">
      <p className="label">{label}</p>
      {payload.map((entry, i) => (
        <p key={i} className="value" style={{ color: entry.color }}>
          {entry.name}: ${fmtCompact(entry.value ?? 0)}
        </p>
      ))}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════
   Main Dashboard Page
   ═══════════════════════════════════════════════════════════════ */

export default function Home() {
  /* ── State ──────────────────────────────────────────────── */
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [historical, setHistorical] = useState<HistoricalResponse | null>(null);
  const [models, setModels] = useState<ModelsListResponse | null>(null);
  const [shap, setShap] = useState<ShapResponse | null>(null);
  const [attention, setAttention] = useState<AttentionResponse | null>(null);

  const [isDemo, setIsDemo] = useState(false);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [backendReady, setBackendReady] = useState(false);
  const [dataStatus, setDataStatus] = useState<DataStatusResponse | null>(null);
  const [range, setRange] = useState<TimeRange>("90d");

  const loadDashboardData = useCallback(async () => {
    setLoading(true);
    const ready = await waitForBackend(30, 2000);
    setBackendReady(ready);

    const [fc, hs, md, sh, at, ds] = await Promise.all([
      fetchJson<ForecastResponse>("/api/v1/forecast"),
      fetchJson<HistoricalResponse>("/api/v1/historical"),
      fetchJson<ModelsListResponse>("/api/v1/models"),
      fetchJson<ShapResponse>("/api/v1/explainability/shap"),
      fetchJson<AttentionResponse>("/api/v1/explainability/attention"),
      fetchJson<DataStatusResponse>("/api/v1/data/status"),
    ]);

    setDataStatus(ds);

    const liveCore = Boolean(fc && hs);
    setIsDemo(!liveCore);
    setForecast(fc ?? MOCK_FORECAST);
    setHistorical(hs ?? MOCK_HISTORICAL);
    setModels(md ?? MOCK_MODELS);
    setShap(sh ?? MOCK_SHAP);
    setAttention(at ?? MOCK_ATTENTION);
    setLoading(false);
  }, []);

  const refreshMarketData = useCallback(async () => {
    setRefreshing(true);
    const result = await postJson<{ status: string; message: string }>(
      "/api/v1/data/refresh"
    );
    if (result) {
      await loadDashboardData();
    }
    setRefreshing(false);
  }, [loadDashboardData]);

  useEffect(() => {
    loadDashboardData();
  }, [loadDashboardData]);

  /* ── Derived data ───────────────────────────────────────── */
  const chartData = useMemo(() => {
    if (!historical) return [];
    const cutoff = RANGE_DAYS[range];
    const sliced = historical.data.slice(-cutoff);

    // append forecast dates
    const combined = sliced.map((p) => ({
      date: p.date,
      actual: p.actual,
      predicted: p.predicted ?? null,
      confLow: null as number | null,
      confHigh: null as number | null,
    }));

    const forecastDatesArray = forecastDates(forecast);
    const forecastPredictionsArray = forecastPredictions(forecast);
    const forecastConfidenceLowArray = forecastConfidenceLow(forecast);
    const forecastConfidenceHighArray = forecastConfidenceHigh(forecast);

    if (forecastDatesArray.length) {
      forecastDatesArray.forEach((d, i) => {
        combined.push({
          date: d,
          actual: null as unknown as number,
          predicted: safeNumber(forecastPredictionsArray[i]),
          confLow: safeNumber(forecastConfidenceLowArray[i]),
          confHigh: safeNumber(forecastConfidenceHighArray[i]),
        });
      });
    }

    return combined;
  }, [historical, forecast, range]);

  const currentPrice = safeNumber(forecast?.current_price, MOCK_FORECAST.current_price);
  const forecastPredictionsArray = forecastPredictions(forecast);
  const firstForecast = safeNumber(forecastPredictionsArray[0]);
  const lastForecast = safeNumber(
    forecastPredictionsArray[forecastPredictionsArray.length - 1]
  );
  const priceChange = firstForecast - currentPrice;
  const pricePct = currentPrice ? (priceChange / currentPrice) * 100 : 0;
  const trendDirection = lastForecast >= currentPrice ? "up" : "down";

  const bestModel = useMemo(() => {
    if (!models?.models.length) return null;
    if (models.best_model) {
      const match = models.models.find((m) => m.name === models.best_model);
      if (match?.metrics?.rmse) return match;
    }
    const ranked = models.models.filter(
      (m) =>
        typeof m.metrics?.rmse === "number" &&
        !Number.isNaN(m.metrics.rmse) &&
        m.metrics.rmse > 0
    );
    if (!ranked.length) return null;
    return ranked.reduce((best, m) =>
      m.metrics.rmse < best.metrics.rmse ? m : best
    );
  }, [models]);

  const hourlyData = useMemo(() => {
    return {
      h1: generateHourlyData(currentPrice),
      h3: generateHourlyData(currentPrice),
      h6: generateHourlyData(currentPrice),
    };
  }, [currentPrice]);

  /* ── Loading skeleton ───────────────────────────────────── */
  if (loading) {
    return (
      <main className="dashboard">
        <header className="header">
          <h1 className="header-brand">Afnan GoldSight AI</h1>
          <p className="header-subtitle">Intelligent Gold Price Forecasting</p>
          <hr className="header-divider" />
        </header>
        <p className="loading-hint">Connecting to API at 127.0.0.1:7860…</p>
        <div className="stats-grid" style={{ marginBottom: 48 }}>
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="glass-card stat-card">
              <div className="skeleton skeleton-text" style={{ margin: "0 auto" }} />
              <div className="skeleton skeleton-value" />
            </div>
          ))}
        </div>
        <div className="glass-card">
          <div className="skeleton skeleton-chart" />
        </div>
      </main>
    );
  }

  /* ═══════════════════════════════════════════════════════════
     Render
     ═══════════════════════════════════════════════════════════ */
  return (
    <main className="dashboard">
      {/* ── Header ──────────────────────────────────────────── */}
      <motion.header
        className="header"
        initial="hidden"
        animate="visible"
        variants={fadeUp}
        transition={{ duration: 0.6 }}
      >
        <h1 className="header-brand">GoldSight AI</h1>
        <p className="header-subtitle">
          Intelligent Gold Price Forecasting — powered by Deep Learning
        </p>
        <hr className="header-divider" />
      </motion.header>

      {/* ── Data freshness ──────────────────────────────────── */}
      {dataStatus && (
        <motion.div
          className={`data-freshness-bar ${dataStatus.days_behind > 2 ? "stale" : ""}`}
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          id="data-freshness-bar"
        >
          <div className="data-freshness-text">
            <strong>Data as of {dataStatus.last_data_date}</strong>
            <span className="data-freshness-meta">
              {dataStatus.source_label} · stored close {fmt(dataStatus.last_close_usd)}
              {dataStatus.live_quote_usd != null && dataStatus.live_quote_date && (
                <>
                  {" "}
                  · Yahoo live ({dataStatus.live_quote_date}):{" "}
                  {fmt(dataStatus.live_quote_usd)}
                </>
              )}
              {dataStatus.days_behind > 0 && (
                <span className="data-stale-hint">
                  {" "}
                  · {dataStatus.days_behind} day{dataStatus.days_behind === 1 ? "" : "s"} behind today
                </span>
              )}
            </span>
          </div>
          <button
            type="button"
            className="data-refresh-btn"
            onClick={() => refreshMarketData()}
            disabled={loading || refreshing}
            id="refresh-market-data-btn"
          >
            {refreshing ? "Downloading…" : "Refresh market data"}
          </button>
        </motion.div>
      )}

      {/* ── Demo Banner ─────────────────────────────────────── */}
      {isDemo && (
        <motion.div
          className="demo-banner"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          id="demo-banner"
        >
          <span className="dot" />
          <span>
            {backendReady
              ? "Could not load live forecast — showing demo data"
              : "Waiting for backend — start it with run.bat or python run_server.py in backend/"}
          </span>
          <button
            type="button"
            className="demo-retry-btn"
            onClick={() => loadDashboardData()}
            disabled={loading}
          >
            {loading ? "Connecting…" : "Retry connection"}
          </button>
        </motion.div>
      )}

      {/* ── Stats Bar ───────────────────────────────────────── */}
      <motion.section
        className="section"
        initial="hidden"
        animate="visible"
        variants={stagger}
      >
        <div className="stats-grid">
          <motion.div className="glass-card stat-card" variants={fadeUp} id="stat-current-price">
            <p className="stat-label">Current Price</p>
            <p className="stat-value gold">{fmt(currentPrice)}</p>
            <p className="stat-sub">
              XAU / USD
              {dataStatus && !isDemo && (
                <> · dataset {dataStatus.last_data_date}</>
              )}
            </p>
          </motion.div>

          <motion.div className="glass-card stat-card" variants={fadeUp} id="stat-24h-change">
            <p className="stat-label">24 h Forecast Δ</p>
            <p className="stat-value">{priceChange >= 0 ? "+" : ""}{fmtCompact(priceChange)}</p>
            <p className={`stat-change ${priceChange >= 0 ? "up" : "down"}`}>
              {priceChange >= 0 ? "▲" : "▼"} {formatNumber(pricePct, 2)}%
            </p>
          </motion.div>

          <motion.div className="glass-card stat-card" variants={fadeUp} id="stat-trend">
            <p className="stat-label">10-Day Trend</p>
            <p className="stat-value" style={{ color: trendDirection === "up" ? "#00e676" : "#ff5252" }}>
              {trendDirection === "up" ? "▲ Bullish" : "▼ Bearish"}
            </p>
            <p className="stat-sub">
              Target: {fmt(lastForecast)}
            </p>
          </motion.div>

          <motion.div className="glass-card stat-card" variants={fadeUp} id="stat-model-accuracy">
            <p className="stat-label">Best Model RMSE</p>
            <p className="stat-value gold">
              {bestModel ? formatNumber(bestModel.metrics.rmse, 2) : "—"}
            </p>
            <p className="stat-sub">{bestModel?.name ?? "N/A"}</p>
          </motion.div>
        </div>
      </motion.section>

      {/* ── Price Chart ─────────────────────────────────────── */}
      <motion.section
        className="section"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, amount: 0.2 }}
        variants={fadeUp}
        transition={{ duration: 0.5 }}
      >
        <div className="glass-card" id="price-chart-section">
          <div className="chart-header">
            <h2 className="section-title">
              <span className="icon">📈</span> Price History &amp; Forecast
            </h2>
            <div className="range-toggle" id="range-toggle">
              {(["30d", "90d", "180d", "365d"] as TimeRange[]).map((r) => (
                <button
                  key={r}
                  id={`range-btn-${r}`}
                  className={`range-btn ${range === r ? "active" : ""}`}
                  onClick={() => setRange(r)}
                >
                  {r}
                </button>
              ))}
            </div>
          </div>

          <div className="chart-container">
            {chartData.length > 0 && (
            <ResponsiveContainer width="100%" height={420}>
              <LineChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis
                  dataKey="date"
                  tick={{ fill: "#6b7280", fontSize: 11 }}
                  tickLine={false}
                  axisLine={{ stroke: "#e0e0e0" }}
                  minTickGap={40}
                />
                <YAxis
                  domain={[
                    (min: number) => Math.floor(min * 0.98),
                    (max: number) => Math.ceil(max * 1.02),
                  ]}
                  tick={{ fill: "#6b7280", fontSize: 11 }}
                  tickLine={false}
                  axisLine={{ stroke: "#e0e0e0" }}
                  tickFormatter={(v: number) => `$${Math.round(v).toLocaleString()}`}
                  width={72}
                />
                <Tooltip content={<ChartTooltip />} />
                <Legend
                  verticalAlign="top"
                  height={36}
                  iconType="plainline"
                  wrapperStyle={{ fontSize: 12, color: "#4b5563" }}
                />
                {/* Actual */}
                <Line
                  type="monotone"
                  dataKey="actual"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  name="Actual"
                  dot={false}
                  activeDot={{ r: 5, stroke: "#3b82f6", strokeWidth: 2, fill: "#ffffff" }}
                  connectNulls={false}
                />
                {/* Predicted */}
                <Line
                  type="monotone"
                  dataKey="predicted"
                  stroke="#ef4444"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  name="Predicted"
                  dot={false}
                  activeDot={{ r: 5, stroke: "#ef4444", strokeWidth: 2, fill: "#ffffff" }}
                  connectNulls={false}
                />
              </LineChart>
            </ResponsiveContainer>
            )}
          </div>
        </div>
      </motion.section>

      {/* ── Hourly Prediction Window ──────────────────────────────── */}
      <motion.section
        className="section"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, amount: 0.2 }}
        variants={fadeUp}
        transition={{ duration: 0.5 }}
      >
        <h2 className="section-title">
          <span className="icon">⏱️</span> Hours Prediction Window
        </h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "16px" }}>
          {[
            { title: "1-Hour Ahead", data: hourlyData.h1 },
            { title: "3-Hours Ahead", data: hourlyData.h3 },
            { title: "6-Hours Ahead", data: hourlyData.h6 }
          ].map((h, idx) => (
            <div key={idx} className="glass-card" style={{ padding: "16px" }}>
              <h3 style={{ fontSize: "1rem", textAlign: "center", marginBottom: "12px", color: "#111827", fontWeight: 600 }}>{h.title}</h3>
              <div style={{ height: "220px" }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={h.data} margin={{ top: 5, right: 5, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                    <XAxis dataKey="time" tick={false} axisLine={{ stroke: "#e0e0e0" }} />
                    <YAxis domain={['auto', 'auto']} tick={{ fill: "#6b7280", fontSize: 10 }} axisLine={{ stroke: "#e0e0e0" }} tickLine={false} />
                    <Tooltip content={<ChartTooltip />} />
                    <Line type="monotone" dataKey="actual" stroke="#3b82f6" strokeWidth={1.5} dot={false} name="Actual" connectNulls={false} />
                    <Line type="monotone" dataKey="predicted" stroke="#ef4444" strokeWidth={1.5} strokeDasharray="4 4" dot={false} name="Predicted" connectNulls={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          ))}
        </div>
      </motion.section>

      {/* ── Forecast Cards ──────────────────────────────────── */}
      {forecast && (
        <motion.section
          className="section"
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.2 }}
          variants={stagger}
        >
          <h2 className="section-title">
            <span className="icon">🔮</span> 10-Day Forecast
          </h2>
          <div className="forecast-grid">
            {forecastDates(forecast).map((d, i) => {
              const prev = i === 0 ? currentPrice : safeNumber(forecastPredictionsArray[i - 1], currentPrice);
              const curr = safeNumber(forecastPredictionsArray[i], currentPrice);
              const up = curr >= prev;
              return (
                <motion.div
                  key={d}
                  className="glass-card forecast-day-card"
                  variants={fadeUp}
                  id={`forecast-card-${i}`}
                >
                  <p className="forecast-date">
                    {new Date(d + "T00:00:00").toLocaleDateString("en-US", {
                      weekday: "short",
                      month: "short",
                      day: "numeric",
                    })}
                  </p>
                  <p className="forecast-price">{fmt(curr)}</p>
                  <span className={`forecast-arrow ${up ? "up" : "down"}`}>
                    {up ? "▲" : "▼"} {Math.abs(curr - prev).toFixed(2)}
                  </span>
                </motion.div>
              );
            })}
          </div>
        </motion.section>
      )}

      {/* ── Model Performance Table ─────────────────────────── */}
      {models && (
        <motion.section
          className="section"
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.2 }}
          variants={fadeUp}
          transition={{ duration: 0.5 }}
        >
          <h2 className="section-title">
            <span className="icon">🏆</span> Model Performance
          </h2>
          <div className="glass-card" id="model-table-section">
            <div className="table-wrapper">
              <table className="data-table" id="model-table">
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>RMSE</th>
                    <th>MAE</th>
                    <th>MAPE</th>
                    <th>R²</th>
                    <th>Dir. Accuracy</th>
                    <th>Checkpoint</th>
                  </tr>
                </thead>
                <tbody>
                  {models.models.map((m, idx) => (
                    <motion.tr
                      key={m.name}
                      initial={{ opacity: 0, x: -20 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      viewport={{ once: true }}
                      transition={{ delay: idx * 0.07 }}
                    >
                      <td className="model-name">{m.name}</td>
                      <td>{formatNumber(m.metrics.rmse, 2)}</td>
                      <td>{formatNumber(m.metrics.mae, 2)}</td>
                      <td>{pctFmt(m.metrics.mape)}</td>
                      <td>{m.metrics.r2 !== undefined ? formatNumber(m.metrics.r2, 3) : "—"}</td>
                      <td>
                        {m.metrics.directional_accuracy !== undefined
                          ? pctFmt(m.metrics.directional_accuracy)
                          : "—"}
                      </td>
                      <td>
                        <span className={`checkpoint-badge ${m.checkpoint_exists ? "yes" : "no"}`}>
                          {m.checkpoint_exists ? "Available" : "Missing"}
                        </span>
                      </td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </motion.section>
      )}

      {/* ── Explainability ──────────────────────────────────── */}
      <motion.section
        className="section"
        initial="hidden"
        whileInView="visible"
        viewport={{ once: true, amount: 0.15 }}
        variants={fadeUp}
        transition={{ duration: 0.5 }}
      >
        <h2 className="section-title">
          <span className="icon">🧠</span> Explainability
        </h2>
        <div className="explain-grid">
          {/* SHAP Feature Importance */}
          <div className="glass-card" id="shap-section">
            <h3 className="section-title" style={{ fontSize: "1.05rem", marginBottom: 12 }}>
              SHAP Feature Importance
            </h3>
            {shap && shap.feature_importance.length > 0 && (
              <div className="shap-chart-wrap">
                <ResponsiveContainer width="100%" height={360}>
                  <BarChart
                    data={[...shap.feature_importance].reverse()}
                    layout="vertical"
                    margin={{ top: 5, right: 20, left: 5, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" horizontal={false} />
                    <XAxis
                      type="number"
                      tick={{ fill: "#6b6b80", fontSize: 11 }}
                      tickLine={false}
                      axisLine={false}
                    />
                    <YAxis
                      type="category"
                      dataKey="feature"
                      tick={{ fill: "#9a9ab0", fontSize: 11 }}
                      tickLine={false}
                      axisLine={false}
                      width={110}
                    />
                    <Tooltip
                      cursor={{ fill: "rgba(212,175,55,0.06)" }}
                      contentStyle={{
                        background: "rgba(14,14,24,0.95)",
                        border: "1px solid rgba(212,175,55,0.15)",
                        borderRadius: 8,
                        fontSize: 13,
                      }}
                      labelStyle={{ color: "#9a9ab0" }}
                      itemStyle={{ color: "#FFD700" }}
                    />
                    <Bar dataKey="importance" radius={[0, 6, 6, 0]} maxBarSize={22}>
                      {shap.feature_importance
                        .slice()
                        .reverse()
                        .map((_, i) => {
                          const t = i / (shap.feature_importance.length - 1);
                          const r = Math.round(184 + t * 71);
                          const g = Math.round(134 + t * 81);
                          const b = Math.round(11 + t * 44);
                          return <Cell key={i} fill={`rgb(${r},${g},${b})`} />;
                        })}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {/* Attention Heatmap */}
          <div className="glass-card" id="attention-section">
            <h3 className="section-title" style={{ fontSize: "1.05rem", marginBottom: 12 }}>
              Temporal Attention Weights
            </h3>
            {attention && (
              <>
                <div className="attention-heatmap">
                  {attention.temporal_importance.map((val, i) => {
                    const safeVal = safeNumber(val);
                    const maxVal = Math.max(...attention.temporal_importance.map(safeNumber));
                    const norm = maxVal > 0 ? safeVal / maxVal : 0;
                    const alpha = 0.15 + norm * 0.85;
                    const bg =
                      norm > 0.7
                        ? `rgba(255, 215, 0, ${alpha})`
                        : norm > 0.4
                        ? `rgba(212, 175, 55, ${alpha})`
                        : `rgba(100, 90, 40, ${alpha})`;
                    return (
                      <div
                        key={i}
                        className="attention-cell"
                        style={{ background: bg }}
                        title={`t-${attention.sequence_length - i}: ${formatNumber(safeVal, 3)}`}
                      >
                        {formatNumber(safeVal, 2)}
                      </div>
                    );
                  })}
                </div>
                <div className="attention-label">
                  <span>← Oldest (t-{attention.sequence_length})</span>
                  <span>Most Recent (t-1) →</span>
                </div>
              </>
            )}
          </div>
        </div>
      </motion.section>

      {/* ── Footer ──────────────────────────────────────────── */}
      <footer className="footer">
        <span>Muhammad Afnan khan</span> · Intelligent Gold Price Forecasting
        {dataStatus && !isDemo && (
          <span style={{ marginLeft: 8, color: "#6b6b80" }}>
            · Gold data: {dataStatus.last_data_date} ({fmt(dataStatus.last_close_usd)})
          </span>
        )}
        {forecast && (
          <span style={{ marginLeft: 8, color: "#6b6b80" }}>
            · Forecast generated: {new Date(forecast.generated_at).toLocaleString()}
          </span>
        )}
      </footer>
    </main>
  );
}
