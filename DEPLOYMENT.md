# Deploying GoldSight AI for free

Two pieces, two platforms:

- **Frontend** (Next.js) → **Vercel**
- **Backend** (FastAPI + torch/xgboost/lightgbm/catboost) → **Hugging Face
  Spaces** (Docker SDK) — its free CPU tier has enough RAM to load all the
  models at once, which Render's free 512MB tier will likely choke on.

## 1. Backend → Hugging Face Spaces

1. Go to huggingface.co → **New Space** → SDK: **Docker** → pick a name
   (e.g. `goldsight-api`), visibility Public (needed for the free tier).
2. Push the contents of this repo's `backend/` folder to the Space's git
   remote, with `Dockerfile` and `README.md` at the **repo root** of the
   Space (they're already in `backend/` here for that reason):

   ```bash
   cd backend
   git init
   git remote add space https://huggingface.co/spaces/<you>/goldsight-api
   git add -A
   git commit -m "Deploy backend"
   git push space main
   ```

   > The trained artifacts (`checkpoints/`, `data/*.parquet`,
   > `data/scalers.joblib`, `logs/metrics_*.json`) must be included in this
   > push — `.gitignore` at the project root has already been updated to
   > stop excluding them. If `backend/` is a subfolder of a bigger repo you
   > push differently, just confirm those files land in the Space.

3. Wait for the build to finish (first build installs torch + the
   boosting libs, a few minutes). Once it's "Running", open
   `https://<you>-goldsight-api.hf.space/health` and confirm it reports
   ready.
4. Note the Space URL — you'll need it for step 2.

   Free-tier note: a public CPU Space stays warm as long as it gets
   occasional traffic; after a while fully idle it can go to sleep and take
   ~30-60s to wake on the next request. That's normal.

## 2. Frontend → Vercel

1. Go to vercel.com → **New Project** → import this repo → set **Root
   Directory** to `frontend/`.
2. Framework preset: Next.js (auto-detected).
3. Add an environment variable:
   - `NEXT_PUBLIC_API_URL` = `https://<you>-goldsight-api.hf.space`
     (no trailing slash — this is read in `frontend/next.config.ts`,
     already wired up to proxy `/api/*` to it).
4. Deploy. Vercel builds and gives you a `*.vercel.app` URL.

## 3. Verify

- Open the Vercel URL, the dashboard should load live data (not the demo
  fallback) within a few seconds — if it doesn't, check the Space is
  "Running" and `/health` returns ready, and check the browser console for
  the actual proxied request URL.
- `POST /api/v1/data/refresh` (via the dashboard's "Refresh market data"
  button) pulls the latest Yahoo `GC=F` price into the Space — this works
  the same in production as it did locally.

## Notes / gotchas

- CORS is already wide open (`allow_origins=["*"]` in
  `backend/app/routes/main.py`), so no change needed there.
- The backend Dockerfile installs a CPU-only torch wheel explicitly, since
  the default PyPI torch wheel pulls CUDA and is too large for the Space's
  free image size budget.
- If you'd rather host the backend on Render instead of Hugging Face: it
  works, but stick to the CPU torch wheel, and be ready for the free
  instance (512MB RAM) to struggle if all 5 models load into memory at
  once — consider lazy-loading only the model `/forecast` actually needs.
