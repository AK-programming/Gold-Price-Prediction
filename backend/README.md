---
title: GoldSight AI Backend
emoji: 📈
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 4.44.1
app_file: run_server.py
pinned: false
---

# GoldSight AI — Backend (Hugging Face Space)

FastAPI service serving the trained gold-price forecasting models
(Transformer, LSTM, XGBoost, LightGBM, CatBoost + a Ridge stacking
ensemble).

This Space is set to the **Gradio** SDK, but no Gradio UI is used — the
frontmatter's `app_file: run_server.py` tells the Space to just run that
script directly with `python run_server.py`, which starts the FastAPI app
with uvicorn on port 7860. This is a way to get a free CPU Space (16GB RAM)
without needing the Docker SDK, which some accounts don't have free access
to. (`Dockerfile` / `.dockerignore` in this folder are kept for reference
if you ever deploy this to a platform that does support Docker for free —
they're not used by this Space as configured.)

- API docs: `/docs`
- Health check: `/health`
- Forecast endpoint: `/api/v1/forecast`

## Deploying this Space

1. Create a new Space on huggingface.co → SDK: **Gradio** → upload the
   contents of this `backend/` folder to the Space repo root.
2. Make sure the deploy branch/commit includes the trained artifacts —
   `checkpoints/`, `data/*.parquet`, `data/scalers.joblib`,
   `logs/metrics_*.json` — these are gitignored for local dev but the API
   can't serve `/forecast` without them.
3. It builds automatically and starts `python run_server.py` (per
   `app_file` above), serving the FastAPI app on port 7860.
4. Once it's live, copy the Space URL (`https://<you>-<space-name>.hf.space`)
   and set it as `NEXT_PUBLIC_API_URL` in the Vercel project for the
   frontend.

No secrets are required for read-only inference; CORS is already open to
any origin in `app/routes/main.py`.
