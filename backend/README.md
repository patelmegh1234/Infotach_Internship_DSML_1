<!-- Megh, Upload Date: 2026-07-14 -->
# Backend

The backend exposes an analyst-facing inference API for AcousticSpace.

## Commands

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
pytest
```

## Important Endpoints

- `GET /health`
- `POST /api/analyze`

## Ownership Map

- Megh: `app/main.py`, `app/api/`, `app/core/`, `app/services/`, Docker integration.
- Shubhangi: `app/audio/`, `scripts/download_dataset.py`, feature validation.
- Fathima: `app/ml/`, `scripts/train_baseline.py`, model tests and metrics.

