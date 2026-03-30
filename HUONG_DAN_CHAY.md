# HƯỚNG DẪN CHẠY VRPTW FULLSTACK

Repo đã chuyển sang fullstack-only (FastAPI + React + Kafka local).
Không còn chạy Streamlit.

## 1) Yêu cầu

- Python 3.9+
- Node.js 18+
- Kafka broker local tại `localhost:9092`

## 2) Cài backend

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Tạo file env từ mẫu:

```bash
cp .env.example .env
```

Các biến môi trường đáng chú ý (backend):

- `OSRM_BASE_URL`, `OSRM_TIMEOUT`, `USE_REAL_ROUTES`
- `REPLAN_COOLDOWN_S`
- `REPLAY_DWELL_SPEEDUP`, `REPLAY_DWELL_STEP_MIN`, `REPLAY_WAIT_TARGET_EVENTS`

Gợi ý cho demo nhanh monitoring:

- `REPLAY_DWELL_SPEEDUP=20` hoặc `40`
- `REPLAY_WAIT_TARGET_EVENTS=120` (giảm xuống nếu muốn waiting chạy nhanh hơn)

## 3) Chạy backend API

```bash
source venv/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## 4) Chạy frontend

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

Mở trình duyệt tại `http://localhost:5173`.

## 5) Kafka local (không dùng Docker trong flow này)

Đảm bảo broker local đang chạy:

- `KAFKA_BOOTSTRAP_SERVERS=localhost:9092`

Backend sẽ tự start Kafka forwarder khi boot.
Nếu Kafka tạm thời không sẵn sàng, backend vẫn chạy và tự retry.

## 6) API chính

- `GET /instances`
- `POST /upload`
- `POST /run`
- `GET /result/{run_id}`
- `POST /compare/quick` (benchmark nhanh cho màn hình kết quả)
- `POST /monitor/start`
- `POST /monitor/stop`
- `GET /monitor/context`
- `POST /monitor/replan`
- `POST /monitor/traffic/inject`
- `WS /ws`

## 7) Gợi ý test nhanh

1. Mở UI -> chọn dataset built-in.
2. Bấm Run -> xem convergence realtime.
3. Khi run complete -> bấm Go to Monitoring.
4. Start replay -> xem vehicle telemetry và alerts.
5. Inject traffic + thử replan (có cooldown 120s mặc định).
6. Quay lại tab Planning -> phần "So sanh nhanh (beta)" -> bấm chạy benchmark:
   - `HGA vs GA thuong vs NN`
   - `traffic dong vs traffic tinh`
   - `co re-plan vs khong re-plan`
   - `monitoring-like vs toi uu tinh`

## 8) Lưu ý về Monitoring/Replay

- Khi xe tới customer sớm hơn `ready_time`, hệ thống vào pha `waiting` (sau đó `servicing`).
- Replay đã phát telemetry theo chunk trong pha waiting/service để đồng hồ và map vẫn cập nhật.
- Tốc độ waiting được tăng riêng bằng `REPLAY_DWELL_SPEEDUP` và điều chỉnh theo từng xe bằng `REPLAY_WAIT_TARGET_EVENTS`.
- Nếu trạng thái run là `complete`, progress bar monitoring sẽ hiển thị `100%`.

