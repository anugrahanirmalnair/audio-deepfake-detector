# Frontend

React + Vite UI for the codec deepfake detector API.

## Run

```bash
cd frontend
npm install
npm run dev
```

Opens **http://localhost:3000**. Proxies `/analyze` and `/health` to the backend at **http://127.0.0.1:8000**.

Start the API first:

```bash
# from repo root
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

## Features

- File upload → `POST /analyze` (proxied)
- Verdict, authenticity / spoof / confidence scores
- Waveform with per-window **recon_score** heatmap (green = low, red = high)
- Window table: `start_ms`, `end_ms`, `recon_score`, `token_score`
