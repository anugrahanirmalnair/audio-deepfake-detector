# Backend API

Codec-based deepfake detection (Random Forest + DAC features).

## Run (from repo root, using project venv)

```bash
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

## Endpoints

- `GET /health` — `{"status": "ok"}`
- `POST /analyze` — multipart field `audio` (wav, flac, mp3, m4a, ogg, webm)

## Example

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/analyze \
  -F "audio=@path/to/sample.flac"
```

Models loaded from `models/deepfake_detector_rf.pkl` and `models/feature_cols.pkl` on startup.
