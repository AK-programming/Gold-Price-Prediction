# GoldSight AI — Gold Price Prediction

Multi-model gold price forecasting dashboard with a FastAPI backend and Next.js frontend.

## Quick start (Windows)

```bat
run.bat
```

Opens:

- Dashboard: http://localhost:3000
- API docs: http://localhost:7860/docs

## First-time model setup

From the `backend` folder:

```bash
cd backend
pip install -r requirements.txt

# Full train (download data + all models + metrics JSON) — can take a while
python train_and_setup.py

# Or only evaluate existing checkpoints and write metrics (faster)
python eval_models.py

# Build stacking ensemble (after all 5 base models exist)
python build_ensemble.py
```

Required artifacts after training:

| File | Purpose |
|------|---------|
| `backend/data/features.parquet` | Engineered features |
| `backend/data/scalers.joblib` | Fitted scaler |
| `backend/checkpoints/*.pt` / `*.joblib` | Model weights |
| `backend/logs/metrics_*.json` | Validation metrics & best model for `/forecast` |

## Manual start

**Backend**

```bash
cd backend
python run_server.py
```

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

The frontend proxies `/api/*` to `http://127.0.0.1:7860` (see `frontend/next.config.ts`). On Windows, use `127.0.0.1` instead of `localhost` to avoid IPv6 connection issues.

If the dashboard shows **demo data**, start the backend first and click **Retry connection**, or wait ~60s for auto-retry.

Use **Refresh market data** on the dashboard (or `POST /api/v1/data/refresh`) to download the latest Yahoo `GC=F` prices. The bar shows **Data as of** the last stored trading day and a **Yahoo live** quote for comparison.

## API smoke tests

```bash
cd backend
pip install pytest httpx
pytest tests/ -v
```

## Project layout

```
backend/
  app/              # FastAPI routes, models, pipeline
  train_and_setup.py
  eval_models.py
  run_server.py
frontend/
  src/app/page.tsx  # Dashboard
run.bat             # One-click dev launcher
```

## Models

- **Deep learning:** Transformer (quantile), LSTM
- **Tree models:** XGBoost, LightGBM, CatBoost
- **Ensemble:** Stacking meta-learner (Ridge)

The `/api/v1/forecast` endpoint serves the model with the **lowest validation RMSE** recorded in the latest `metrics_*.json` file.

## Health check

`GET /health` returns readiness for data, scaler, checkpoints, and metrics.
