import { useEffect, useMemo, useState } from "react";
import { useMonitoring } from "../context/MonitoringContext";
import { useRun } from "../context/RunContext";
import AlertFeed from "./AlertFeed";
import RouteMap from "./RouteMap";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const TELEMETRY_STALE_MS = 15000;

function formatClockMinutes(m) {
  if (m == null || Number.isNaN(Number(m))) return "—";
  const m24 = ((Number(m) % 1440) + 1440) % 1440;
  const hh = Math.floor(m24 / 60);
  const mm = Math.round(m24 % 60);
  return `${String(hh).padStart(2, "0")}:${String(mm).padStart(2, "0")}`;
}

export default function MonitoringView() {
  const { state, startMonitor, stopMonitor, requestReplan, injectTraffic } = useMonitoring();
  const { state: runState } = useRun();
  const [contextData, setContextData] = useState(null);
  const [nowMs, setNowMs] = useState(Date.now());
  const [speedX, setSpeedX] = useState(1);
  const [trafficFactorInput, setTrafficFactorInput] = useState(1.2);
  const [trafficFromH, setTrafficFromH] = useState(0);
  const [trafficToH, setTrafficToH] = useState(1e6);
  const [trafficLabel, setTrafficLabel] = useState("manual_override");
  const [simAnchorH, setSimAnchorH] = useState(0);
  const [wallAnchorMs, setWallAnchorMs] = useState(Date.now());

  useEffect(() => {
    if (!state.monitorRunId) return;
    fetch(`${API_URL}/monitor/context?run_id=${state.monitorRunId}&replay_slot=${state.replaySlot}`)
      .then((r) => r.json())
      .then(setContextData)
      .catch(() => setContextData(null));
  }, [state.monitorRunId, state.replaySlot, state.planRevision]);

  useEffect(() => {
    const t = setInterval(() => setNowMs(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    setSimAnchorH(Number(state.simTimeH || 0));
    setWallAnchorMs(Date.now());
  }, [state.simTimeH]);

  const vehicles = Object.values(state.telemetry || {});

  const latestTelemetry = useMemo(() => {
    if (!vehicles.length) return null;
    return vehicles.reduce((a, b) => ((a.sim_time_h || 0) >= (b.sim_time_h || 0) ? a : b));
  }, [vehicles]);

  if (!state.monitorRunId) return <div style={{ color: "#64748b" }}>Arm monitor from Planning first.</div>;

  const cooldownMs = Math.max(0, (state.replanCooldownUntilMs || 0) - nowMs);
  const cooldownSec = Math.ceil(cooldownMs / 1000);
  const canReplan = state.monitoringState === "simulating" && cooldownMs === 0;
  const telemetryStale =
    state.monitoringState === "simulating" &&
    state.lastTelemetryWallMs != null &&
    nowMs - state.lastTelemetryWallMs > TELEMETRY_STALE_MS;

  const replaySpeed = Number(state.speedX) > 0 ? Number(state.speedX) : speedX;

  const simClockH =
    state.monitoringState === "simulating"
      ? telemetryStale
        ? simAnchorH
        : simAnchorH + ((nowMs - wallAnchorMs) / 1000) * (replaySpeed / 3600)
      : Number(state.simTimeH || 0);
  const replayDurationH = Number(contextData?.replay_duration_h || 0);
  const progressPct =
    state.monitoringState === "complete"
      ? 100
      : replayDurationH > 0
        ? Math.min(100, (simClockH / replayDurationH) * 100)
        : 0;

  const clockMinutes =
    contextData?.start_minutes != null
      ? Number(contextData.start_minutes) + simClockH * 60
      : state.lastClockMinutes != null
        ? state.lastClockMinutes
        : null;

  const clockLabel = formatClockMinutes(clockMinutes);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr minmax(300px, 360px)", gap: 16, alignItems: "start" }}>
      <div
        style={{
          borderRadius: 12,
          overflow: "hidden",
          border: "1px solid #e2e8f0",
          boxShadow: "0 2px 8px rgba(15,23,42,0.06)",
          minWidth: 0,
        }}
      >
        <RouteMap
          depot={contextData?.depot}
          customers={contextData?.customers}
          telemetry={state.telemetry}
          trails={state.trails}
          polylines={contextData?.polylines}
          alerts={state.alerts}
          mapHeight={520}
          variant="replay"
          visitedCustomerIds={state.visitedCustomerIds}
          customerVehicleIndex={contextData?.customer_vehicle_index}
          depotCompact
        />
      </div>

      <div
        style={{
          display: "grid",
          gap: 12,
          padding: 16,
          background: "#f8fafc",
          borderRadius: 12,
          border: "1px solid #e2e8f0",
          fontSize: 13,
          color: "#334155",
        }}
      >
        <div
          style={{
            padding: 14,
            background: "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)",
            borderRadius: 10,
            color: "#f8fafc",
          }}
        >
          <div style={{ fontSize: 11, opacity: 0.85, letterSpacing: "0.06em", textTransform: "uppercase" }}>
            Giờ mô phỏng (đồng bộ bảng Planning)
          </div>
          <div style={{ fontSize: 32, fontWeight: 700, fontVariantNumeric: "tabular-nums", marginTop: 6 }}>{clockLabel}</div>
          <div style={{ fontSize: 11, opacity: 0.75, marginTop: 8 }}>
            Cùng hệ quy chiếu phút trong ngày với cột “Đến / Bắt đầu” ở Planning (khi replay chạy, ưu tiên số từ server).
          </div>
        </div>

        <div style={{ padding: 12, background: "#fff", borderRadius: 8, border: "1px solid #e2e8f0" }}>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>Trạng thái</div>
          <div>Run: {state.monitoringState}</div>
          <div style={{ marginTop: 4 }}>Sim đã chạy: {simClockH.toFixed(2)} h (trục nội bộ)</div>
          <div style={{ marginTop: 4 }}>Revision: {state.planRevision}</div>
        </div>

        <div style={{ padding: 12, background: "#fff", borderRadius: 8, border: "1px solid #e2e8f0" }}>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>Tắc nghẽn (theo đồng hồ)</div>
          <div>
            Mô hình: {runState.trafficModel} — hệ số hiện tại ~{Number(latestTelemetry?.traffic_factor ?? state.trafficFactor).toFixed(2)} (
            {latestTelemetry?.traffic_source ?? state.trafficSource})
          </div>
          <div style={{ fontSize: 12, color: "#64748b", marginTop: 6 }}>
            Adaptive đã tính theo phút trong ngày khi replay; không cần chèn tay trừ khi muốn ghi đè.
          </div>
        </div>

        <div style={{ padding: 12, background: "#fff", borderRadius: 8, border: "1px solid #e2e8f0" }}>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>Điều khiển replay</div>
          <label style={{ display: "grid", gap: 4, marginBottom: 8 }}>
            Cấp tốc
            <select value={speedX} onChange={(e) => setSpeedX(Number(e.target.value))}>
              <option value={1}>1x</option>
              <option value={10}>10x</option>
              <option value={60}>60x</option>
            </select>
          </label>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            <button
              type="button"
              onClick={() => startMonitor(state.monitorRunId, state.replaySlot, speedX).catch(() => undefined)}
            >
              Bắt đầu replay
            </button>
            <button
              type="button"
              onClick={() => stopMonitor(state.monitorRunId).catch(() => undefined)}
            >
              Dừng
            </button>
          </div>
          <p style={{ margin: "8px 0 0", fontSize: 12, color: "#64748b", lineHeight: 1.45 }}>
            <strong>Re-plan</strong>: chạy lại GA ngắn trên cùng bài toán để có lộ trình mới (khi tắc đổi mạnh hoặc
            muốn thử phương án khác). Có cooldown ~2 phút; không bắt buộc khi chỉ xem replay.
          </p>
          <button
            type="button"
            style={{ marginTop: 8, width: "100%" }}
            disabled={!canReplan}
            onClick={() =>
              requestReplan(
                state.monitorRunId,
                0,
                state.monitoringState === "simulating" ? simClockH : Number(state.simTimeH || 0)
              ).catch(() => undefined)
            }
          >
            {canReplan ? "Re-plan (chạy lại GA)" : `Re-plan sau ${cooldownSec}s`}
          </button>
        </div>

        <details style={{ padding: 12, background: "#fff", borderRadius: 8, border: "1px solid #e2e8f0" }}>
          <summary style={{ cursor: "pointer", fontWeight: 600 }}>Ghi đè tắc (tùy chọn)</summary>
          <div style={{ display: "grid", gap: 8, marginTop: 10 }}>
            <label>
              Hệ số
              <input
                type="number"
                min={0.3}
                max={5}
                step={0.1}
                value={trafficFactorInput}
                onChange={(e) => setTrafficFactorInput(Number(e.target.value))}
              />
            </label>
            <label>
              từ h
              <input type="number" min={0} step={0.1} value={trafficFromH} onChange={(e) => setTrafficFromH(Number(e.target.value))} />
            </label>
            <label>
              đến h
              <input type="number" min={0} step={0.1} value={trafficToH} onChange={(e) => setTrafficToH(Number(e.target.value))} />
            </label>
            <label>
              nhãn
              <input value={trafficLabel} onChange={(e) => setTrafficLabel(e.target.value)} />
            </label>
            <button type="button" onClick={() => injectTraffic(state.monitorRunId, trafficFactorInput, trafficFromH, trafficToH, trafficLabel)}>
              Áp dụng ghi đè
            </button>
          </div>
        </details>

        <div style={{ height: 8, background: "#e2e8f0", borderRadius: 4 }}>
          <div style={{ width: `${progressPct}%`, height: "100%", background: "#4f46e5", borderRadius: 4 }} />
        </div>
        <div style={{ fontSize: 12, color: "#64748b" }}>
          Tiến độ: {progressPct.toFixed(1)}% ({simClockH.toFixed(2)} h / {replayDurationH.toFixed(2)} h ước lượng)
        </div>
        {telemetryStale ? (
          <div style={{ fontSize: 12, color: "#b45309", background: "#fffbeb", padding: 8, borderRadius: 6 }}>
            Hơn {Math.round(TELEMETRY_STALE_MS / 1000)} giây không nhận telemetry từ server — bản đồ và đồng hồ giữ
            theo lần cập nhật cuối; thanh tiến độ không còn tự chạy. Replay có thể đang chờ OSRM hoặc bị kẹt.
          </div>
        ) : null}

        <div style={{ maxHeight: 200, overflow: "auto", border: "1px solid #e2e8f0", borderRadius: 8, padding: 8, background: "#fff" }}>
          {vehicles.length === 0 ? (
            <div style={{ color: "#94a3b8" }}>Chưa có telemetry.</div>
          ) : (
            vehicles.map((v) => {
              const delay = Number(v.eta_h || 0) - Number(v.planned_arrival_h || 0);
              return (
                <div
                  key={v.vehicle_id}
                  style={{
                    fontSize: 12,
                    fontFamily: "ui-monospace, monospace",
                    padding: "4px 0",
                    borderBottom: "1px solid #f1f5f9",
                    color: v.status === "done" ? "#15803d" : delay > 0.05 ? "#b91c1c" : "#334155",
                  }}
                >
                  V{v.vehicle_id} | {v.status}
                  {v.status === "waiting" || v.status === "servicing" ? ` | at=${v.next_customer_id} | rem=${v.dwell_remaining_minutes != null ? Number(v.dwell_remaining_minutes).toFixed(1) + "m" : "—"}` : ""}
                  {v.status === "arrived" ? ` | arrived=${v.next_customer_id}` : ""}
                  {v.status !== "waiting" && v.status !== "servicing" && v.status !== "arrived" ? ` | next=${v.next_customer_id}` : ""}
                  | eta={Number(v.eta_h || 0).toFixed(2)} | v={v.speed_kmh != null ? `${v.speed_kmh} km/h` : "—"}
                </div>
              );
            })
          )}
        </div>
        <AlertFeed alerts={state.alerts} />
      </div>
    </div>
  );
}
