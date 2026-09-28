# GoldSight AI — backend image for Hugging Face Spaces (Docker SDK)
FROM python:3.11-slim

WORKDIR /app

# System deps needed by lightgbm/catboost/xgboost wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first (the default PyPI wheel pulls in CUDA and blows past
# the free Space's image size limit) — pin to a version that satisfies
# requirements.txt's torch>=2.0.0 so the later install is a no-op.
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + the trained artifacts the API serves at runtime.
# (These are normally .gitignored for local dev — make sure your deploy
# branch/commit actually includes checkpoints/, data/*.parquet,
# data/scalers.joblib and logs/metrics_*.json, or /health and /forecast
# will report not-ready.)
COPY . .

# Hugging Face Spaces (Docker SDK) expects the app on 7860.
ENV API_HOST=0.0.0.0
ENV API_PORT=7860
EXPOSE 7860

CMD ["python", "run_server.py"]
