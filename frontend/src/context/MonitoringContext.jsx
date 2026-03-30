import { createContext, useContext, useMemo, useReducer } from "react";

const MonitoringContext = createContext(null);
const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const TRAIL_MAX_POINTS = 4000;

const initialState = {
  monitorRunId: null,
  monitoringState: "idle",
  telemetry: {},
  trails: {},
  alerts: [],
  simTimeH: 0,
  replaySlot: 0,
  speedX: 1,
  planRevision: 0,
  trafficFactor: 1.0,
  trafficSource: "manual_override",
  trafficLabel: "manual_override",
  trafficFromH: 0,
  trafficToH: 1e6,
  replanCooldownUntilMs: 0,
  error: null,
  visitedCustomerIds: [],
  lastClockMinutes: null,
  lastTelemetryWallMs: null,
};

function reducer(state, action) {
  switch (action.type) {
    case "ARM_MONITOR":
      return { ...state, monitorRunId: action.payload };
    case "MON_API_STARTED":
      return {
        ...state,
        monitoringState: "simulating",
        visitedCustomerIds: [],
        trails: {},
        telemetry: {},
        lastClockMinutes: null,
        lastTelemetryWallMs: null,
        simTimeH: 0,
        alerts: [],
      };
    case "MONITOR_STOP_LOCAL":
      return {
        ...state,
        monitoringState: "idle",
        telemetry: {},
        lastClockMinutes: null,
        lastTelemetryWallMs: null,
      };
    case "SET_REPLAY_SLOT":
      return { ...state, replaySlot: action.payload };
    case "SET_SPEED_X":
      return { ...state, speedX: action.payload };
    case "WS_MESSAGE": {
      const msg = action.payload || {};
      const receivedAt = action.receivedAt;
      if (!state.monitorRunId || msg.run_id !== state.monitorRunId) {
        return state;
      }
      if (msg.type === "telemetry") {
        const key = String(msg.vehicle_id);
        const prev = state.telemetry[key];
        const visited = new Set(state.visitedCustomerIds);
        if (msg.arrived_customer_id != null && Number(msg.arrived_customer_id) !== 0) {
          visited.add(Number(msg.arrived_customer_id));
        }
        if (prev && prev.next_customer_id != null && msg.next_customer_id !== prev.next_customer_id) {
          if (prev.next_customer_id !== 0) {
            visited.add(prev.next_customer_id);
          }
        }
        const prevTrail = state.trails[key] || [];
        const lastTrailPoint = prevTrail.length ? prevTrail[prevTrail.length - 1] : null;
        const shouldAppendTrail =
          !lastTrailPoint ||
          Number(lastTrailPoint[0]) !== Number(msg.lat) ||
          Number(lastTrailPoint[1]) !== Number(msg.lon);
        const nextTrail = shouldAppendTrail ? [...prevTrail, [msg.lat, msg.lon]].slice(-TRAIL_MAX_POINTS) : prevTrail;
        return {
          ...state,
          telemetry: { ...state.telemetry, [key]: msg },
          trails: { ...state.trails, [key]: nextTrail },
          simTimeH: Math.max(
            Number(state.simTimeH || 0),
            Number(msg.sim_time_h != null ? msg.sim_time_h : 0)
          ),
          visitedCustomerIds: Array.from(visited),
          lastClockMinutes: msg.clock_minutes != null ? msg.clock_minutes : state.lastClockMinutes,
          lastTelemetryWallMs: receivedAt ?? state.lastTelemetryWallMs,
        };
      }
      if (msg.type === "alert") {
        return { ...state, alerts: [msg.data, ...state.alerts].slice(0, 50) };
      }
      if (msg.type === "sim_complete") {
        return {
          ...state,
          monitoringState: msg.cancelled ? "idle" : "complete",
        };
      }
      if (msg.type === "monitor_stopped") {
        return {
          ...state,
          monitoringState: "idle",
          telemetry: {},
          lastClockMinutes: null,
          lastTelemetryWallMs: null,
        };
      }
      if (msg.type === "monitor_error" || msg.type === "replan_error") {
        return { ...state, error: msg.message || "Monitoring error" };
      }
      if (msg.type === "replan_started") {
        return {
          ...state,
          monitoringState: "replanning",
          replanCooldownUntilMs: Date.now() + 120000,
        };
      }
      if (msg.type === "replan_complete") {
        return {
          ...state,
          monitoringState: "simulating",
          planRevision: msg.plan_revision || state.planRevision,
        };
      }
      if (msg.type === "traffic_update") {
        return {
          ...state,
          trafficFactor: msg.factor ?? state.trafficFactor,
          trafficSource: msg.source ?? state.trafficSource,
          trafficLabel: msg.label ?? state.trafficLabel,
          trafficFromH: msg.from_h ?? state.trafficFromH,
          trafficToH: msg.to_h ?? state.trafficToH,
        };
      }
      return state;
    }
    default:
      return state;
  }
}

export function MonitoringProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const api = useMemo(
    () => ({
      dispatch,
      async startMonitor(runId, replaySlot = 0, speed = 1) {
        dispatch({ type: "ARM_MONITOR", payload: runId });
        dispatch({ type: "SET_REPLAY_SLOT", payload: replaySlot });
        dispatch({ type: "SET_SPEED_X", payload: speed });
        const resp = await fetch(`${API_URL}/monitor/start`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            run_id: runId,
            replay_slot: replaySlot,
            hours_per_real_second: speed,
          }),
        });
        if (!resp.ok) {
          throw new Error("Failed to start monitor");
        }
        dispatch({ type: "MON_API_STARTED" });
      },
      async stopMonitor(runId) {
        await fetch(`${API_URL}/monitor/stop`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ run_id: runId }),
        });
        dispatch({ type: "MONITOR_STOP_LOCAL" });
      },
      async injectTraffic(runId, factor, fromH, toH, label) {
        await fetch(`${API_URL}/monitor/traffic/inject`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            run_id: runId,
            factor,
            from_h: fromH,
            to_h: toH,
            label,
          }),
        });
      },
      async requestReplan(runId, replaySlot, simTimeH) {
        const resp = await fetch(`${API_URL}/monitor/replan`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ run_id: runId, replay_slot: replaySlot, sim_time_h: simTimeH }),
        });
        if (!resp.ok) {
          const data = await resp.json().catch(() => ({}));
          throw new Error(data.detail || "Failed to request replan");
        }
      },
    }),
    []
  );
  return <MonitoringContext.Provider value={{ state, ...api }}>{children}</MonitoringContext.Provider>;
}

export function useMonitoring() {
  const ctx = useContext(MonitoringContext);
  if (!ctx) {
    throw new Error("useMonitoring must be used within MonitoringProvider");
  }
  return ctx;
}
