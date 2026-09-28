---
title: GoldSight AI Backend
emoji: 📈
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# GoldSight AI — Backend (Hugging Face Space)

FastAPI service serving the trained gold-price forecasting models
(Transformer, LSTM, XGBoost, LightGBM, CatBoost + a Ridge stacking
ensemble). This Space builds the `Dockerfile` in this folder and exposes
the API on port 7860.

- API docs: `/docs`
- Health check: `/health`
- Forecast endpoint: `/api/v1/forecast`

## Deploying this Space

1. Create a new Space on huggingface.co → SDK: **Docker** → link it to a
   repo containing the contents of this `backend/` folder at the repo root
   (Dockerfile included).
2. Make sure the deploy branch/commit includes the trained artifacts —
   `checkpoints/`, `data/*.parquet`, `data/scalers.joblib`,
   `logs/metrics_*.json` — these are gitignored for local dev but the API
   can't serve `/forecast` without them.
3. Push. The Space builds the Dockerfile and starts `python run_server.py`.
4. Once it's live, copy the Space URL (`https://<you>-<space-name>.hf.space`)
   and set it as `NEXT_PUBLIC_API_URL` in the Vercel project for the
   frontend.

No secrets are required for read-only inference; CORS is already open to
any origin in `app/routes/main.py`.
