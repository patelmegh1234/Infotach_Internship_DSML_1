<!-- Megh, Upload Date: 2026-07-14 -->
# AcousticSpace

AcousticSpace is a mid-review ready deepfake audio detection project that focuses on room acoustics, reverberation, background consistency, and breathing cadence proxies instead of only vocal artifacts.

The Week 1-2 version includes a FastAPI backend, a Librosa-based acoustic feature pipeline, a baseline confidence model, a React analyst dashboard, waveform visualization, project documentation, Docker files, and a team contribution plan.

## Team Ownership

| Member | Primary Area | Upload Focus |
| --- | --- | --- |
| Megh | Repository owner, API integration, coordination | Backend API, Docker, CI, deployment docs, merge reviews |
| Shubhangi | Audio processing pipeline | Dataset setup, Librosa features, spectrogram and RIR extraction |
| Fathima | Model development | Baseline classifier, training scripts, evaluation metrics |
| Dimple | Frontend dashboard | React UI, upload flow, waveform visualization, results history |

Every member should commit their assigned files from their own GitHub account so senior reviewers can see real contribution history.

## Mid-Project Review Scope

This repository intentionally focuses on the first two weeks of the plan:

- FastAPI server with upload and inference endpoints.
- Dataset download helper for `awsaf49/asvpoof-2019-dataset` through `kagglehub`.
- Librosa feature extraction for spectrogram, MFCC, reverb tail, RIR-style decay proxy, background noise consistency, and breathing cadence proxy.
- Baseline acoustic mismatch model with confidence scores and suspicious segment output.
- React dashboard with upload, waveform preview, result cards, anomaly table, and local analysis history.
- Docker and CI scaffolding for deployment readiness.

Week 3-4 placeholders are included in the docs and model interfaces so the team can later replace the baseline model with an AST fine-tuning pipeline.

## Quick Start

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` for the API contract.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

### Docker Compose

```bash
docker compose up --build
```

Frontend: `http://localhost:5173`

Backend: `http://localhost:8000`

## Dataset Setup

The ASVspoof dataset is large, so the repository includes a download helper instead of committing dataset files.

```bash
cd backend
python scripts/download_dataset.py
```

The script uses:

```python
import kagglehub
path = kagglehub.dataset_download("awsaf49/asvpoof-2019-dataset")
```

Store local data outside git under `data/raw/`.

## Project Structure

```text
backend/                 FastAPI, audio processing, baseline model
frontend/                React analyst dashboard
docs/                    Architecture, dataset, review, API, team workflow
deployment/              Deployment notes and production checklist
.github/workflows/       CI checks
```

## Review Demo Flow

1. Start backend and frontend.
2. Upload a WAV/MP3/FLAC audio clip in the dashboard.
3. Show waveform rendering.
4. Submit for analysis.
5. Explain the acoustic feature cards: reverb tail, RIR decay proxy, background consistency, breathing cadence proxy.
6. Show confidence output and suspicious segment table.
7. Open `docs/MID_PROJECT_REVIEW.md` to explain Week 1-2 completion and Week 3-4 plan.

