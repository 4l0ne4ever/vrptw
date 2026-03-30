# VRPTW Fullstack (FastAPI + React + Kafka)

This repository now runs in fullstack mode only (no Streamlit fallback):

- Backend API + WebSocket: `backend/main.py`
- Frontend SPA (Vite + React): `frontend/`
- Solver core: `src/`
- Messaging/replay: `src/messaging`, `src/simulation`

## Quick Start

### 1) Python backend

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### 2) Local Kafka broker

Use a locally running Kafka broker at `localhost:9092`.

Default env:

```bash
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
```

See `.env.example` for all backend env keys.

### 3) Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## API Endpoints

- `GET /health`
- `GET /instances`
- `POST /upload`
- `POST /run`
- `GET /result/{run_id}`
- `POST /compare/quick`
- `POST /monitor/start`
- `POST /monitor/stop`
- `GET /monitor/context`
- `POST /monitor/replan`
- `POST /monitor/traffic/inject`
- `WS /ws`

## Notes

- Realtime flow: solver/replay emit Kafka -> backend forwarder -> websocket broadcast.
- Replan supports cooldown (`REPLAN_COOLDOWN_S`).
- Quick benchmark endpoint (`/compare/quick`) provides lightweight comparisons:
  - HGA vs GA vs NN
  - adaptive traffic vs static traffic
  - re-plan vs no re-plan (actual when available, otherwise quick simulated pair)
  - monitoring-like vs static optimization
- Run artifacts are stored in `results/runs/<run_id>/artifacts.pkl`.
