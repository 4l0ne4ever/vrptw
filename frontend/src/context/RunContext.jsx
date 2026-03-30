import { createContext, useContext, useMemo, useReducer } from "react";

const RunContext = createContext(null);
const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const initialState = {
  runState: "idle",
  currentRunId: null,
  convergence: [],
  result: null,
  comparison: null,
  comparisonState: "idle",
  comparisonError: null,
  solverProgressMessage: "",
  error: null,
  trafficModel: "adaptive",
  instances: [],
};

function reducer(state, action) {
  switch (action.type) {
    case "RESET":
      return { ...initialState, instances: state.instances };
    case "SET_INSTANCES":
      return { ...state, instances: action.payload || [] };
    case "RUN_STARTED":
      return {
        ...state,
        runState: "running",
        currentRunId: action.payload,
        convergence: [],
        result: null,
        comparison: null,
        comparisonState: "idle",
        comparisonError: null,
        error: null,
      };
    case "RESULT_LOADED":
      return { ...state, runState: "complete", result: action.payload };
    case "RUN_PROGRESS": {
      const incoming = action.payload || {};
      const byGen = new Map();
      for (const item of [...state.convergence, ...(incoming.convergence || [])]) {
        if (item?.generation == null) continue;
        byGen.set(Number(item.generation), item);
      }
      const merged = Array.from(byGen.values()).sort((a, b) => Number(a.generation) - Number(b.generation));
      const nextRunState =
        incoming.status === "pending" ||
        incoming.status === "running" ||
        incoming.status === "complete" ||
        incoming.status === "error"
          ? incoming.status
          : state.runState;
      return {
        ...state,
        runState: nextRunState,
        convergence: merged,
        solverProgressMessage: incoming.solver_progress_message ?? state.solverProgressMessage,
      };
    }
    case "SET_ERROR":
      return { ...state, runState: "error", error: action.payload || "Unknown error" };
    case "SET_TRAFFIC_MODEL":
      return { ...state, trafficModel: action.payload };
    case "COMPARE_STARTED":
      return { ...state, comparisonState: "running", comparisonError: null };
    case "COMPARE_LOADED":
      return { ...state, comparisonState: "complete", comparison: action.payload, comparisonError: null };
    case "COMPARE_ERROR":
      return { ...state, comparisonState: "error", comparisonError: action.payload || "Comparison failed" };
    case "WS_MESSAGE": {
      const msg = action.payload || {};
      if (!state.currentRunId || msg.run_id !== state.currentRunId) {
        return state;
      }
      if (msg.type === "convergence") {
        return {
          ...state,
          convergence: [
            ...state.convergence,
            {
              generation: msg.generation,
              best_fitness: msg.best_fitness,
              avg_fitness: msg.avg_fitness,
            },
          ],
        };
      }
      if (msg.type === "solver_progress") {
        return { ...state, solverProgressMessage: msg.message || "" };
      }
      if (msg.type === "run_complete") {
        return { ...state, runState: "complete" };
      }
      if (msg.type === "run_error") {
        return { ...state, runState: "error", error: msg.message || "Run failed" };
      }
      return state;
    }
    default:
      return state;
  }
}

export function RunProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState);

  const api = useMemo(
    () => ({
      dispatch,
      async fetchInstances() {
        const resp = await fetch(`${API_URL}/instances`);
        const data = await resp.json();
        const items = (data.items || []).filter((it) => it.dataset_type === "test");
        dispatch({ type: "SET_INSTANCES", payload: items });
      },
      async startRun(payload) {
        const resp = await fetch(`${API_URL}/run`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!resp.ok) {
          throw new Error("Failed to start run");
        }
        const data = await resp.json();
        dispatch({ type: "RUN_STARTED", payload: data.run_id });
        return data.run_id;
      },
      async pollResult(runId) {
        const resp = await fetch(`${API_URL}/result/${runId}`);
        const data = await resp.json();
        if (data.status === "complete") {
          dispatch({ type: "RESULT_LOADED", payload: data.result });
        }
        return data;
      },
      async pollProgress(runId) {
        const resp = await fetch(`${API_URL}/progress/${runId}`);
        if (!resp.ok) return null;
        const data = await resp.json();
        dispatch({ type: "RUN_PROGRESS", payload: data });
        return data;
      },
      async uploadFile(file) {
        const form = new FormData();
        form.append("file", file);
        const resp = await fetch(`${API_URL}/upload`, { method: "POST", body: form });
        if (!resp.ok) {
          throw new Error("Upload failed");
        }
        return await resp.json();
      },
      async runQuickComparison(runId) {
        dispatch({ type: "COMPARE_STARTED" });
        const resp = await fetch(`${API_URL}/compare/quick`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ run_id: runId }),
        });
        if (!resp.ok) {
          const data = await resp.json().catch(() => ({}));
          const msg = data.detail || "Failed to run comparison";
          dispatch({ type: "COMPARE_ERROR", payload: msg });
          throw new Error(msg);
        }
        const data = await resp.json();
        dispatch({ type: "COMPARE_LOADED", payload: data.comparison || null });
        return data.comparison;
      },
    }),
    []
  );

  return <RunContext.Provider value={{ state, ...api }}>{children}</RunContext.Provider>;
}

export function useRun() {
  const ctx = useContext(RunContext);
  if (!ctx) {
    throw new Error("useRun must be used within RunProvider");
  }
  return ctx;
}

