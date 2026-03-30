#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PORT="${E2E_PORT:-8010}"
HOST="127.0.0.1"
BASE_URL="http://${HOST}:${PORT}"
LOG_FILE="$(mktemp)"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
  rm -f "$LOG_FILE"
}
trap cleanup EXIT

if [[ ! -f "frontend/dist/index.html" ]]; then
  echo "Missing frontend/dist/index.html. Run: cd frontend && npm run build"
  exit 1
fi

python3 -m uvicorn backend.main:app --host "$HOST" --port "$PORT" >"$LOG_FILE" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 50); do
  if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.2
done

curl -fsS "${BASE_URL}/health" >/dev/null
curl -fsS "${BASE_URL}/instances" >/dev/null
curl -fsS "${BASE_URL}/" >/dev/null

echo "E2E smoke passed: backend API + frontend static mount reachable."
