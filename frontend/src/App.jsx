import { useEffect, useState } from "react";
import ConvergenceChart from "./components/ConvergenceChart";
import MonitoringView from "./components/MonitoringView";
import PlanningMap from "./components/PlanningMap";
import ResultDetailPanel from "./components/ResultDetailPanel";
import ResultMetrics from "./components/ResultMetrics";
import RunControls from "./components/RunControls";
import WebSocketBridge from "./components/WebSocketBridge";
import { useMonitoring } from "./context/MonitoringContext";
import { useRun } from "./context/RunContext";

function PlanningTab({ onGoMonitoring }) {
  const { state, dispatch, fetchInstances, startRun, pollResult, pollProgress, uploadFile, runQuickComparison } = useRun();

  useEffect(() => {
    fetchInstances();
  }, [fetchInstances]);

  useEffect(() => {
    if (state.runState !== "complete" || !state.currentRunId || state.result) return;
    const timer = setInterval(async () => {
      const data = await pollResult(state.currentRunId);
      if (data.status === "complete") {
        clearInterval(timer);
      }
    }, 1500);
    return () => clearInterval(timer);
  }, [state.runState, state.currentRunId, state.result, pollResult]);

  useEffect(() => {
    if (state.runState !== "running" || !state.currentRunId) return;
    const timer = setInterval(async () => {
      await pollProgress(state.currentRunId);
    }, 1000);
    return () => clearInterval(timer);
  }, [state.runState, state.currentRunId, pollProgress]);

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "minmax(240px, 300px) minmax(0, 1fr)",
        gap: 24,
        alignItems: "start",
        width: "100%",
        maxWidth: "100%",
        boxSizing: "border-box",
      }}
    >
      <aside style={{ minWidth: 0 }}>
      <RunControls
        instances={state.instances}
        busy={state.runState === "running"}
        onRun={async (payload) => {
          try {
            await startRun(payload);
          } catch (e) {
            dispatch({ type: "SET_ERROR", payload: e?.message || "Failed to start run" });
          }
        }}
        onUpload={uploadFile}
        onRefreshInstances={fetchInstances}
        runState={state.runState}
        trafficModel={state.trafficModel}
        onTrafficModelChange={(model) => dispatch({ type: "SET_TRAFFIC_MODEL", payload: model })}
        err={state.error}
        onGoMonitoring={onGoMonitoring}
      />
      </aside>
      <main
        style={{
          display: "grid",
          gap: 20,
          minWidth: 0,
          padding: 18,
          background: "#f1f5f9",
          borderRadius: 12,
          border: "1px solid #e2e8f0",
        }}
      >
        <div
          style={{
            fontSize: 14,
            color: "#334155",
            padding: "10px 12px",
            background: "#fff",
            borderRadius: 8,
            border: "1px solid #e2e8f0",
          }}
        >
          Trạng thái: <strong>{state.runState}</strong>
          {state.solverProgressMessage ? ` — ${state.solverProgressMessage}` : ""}
        </div>
        <div
          style={{
            padding: 14,
            background: "#fff",
            borderRadius: 12,
            border: "1px solid #e2e8f0",
            minWidth: 0,
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: 10, color: "#0f172a" }}>Tiến trình tối ưu</div>
          <ConvergenceChart data={state.convergence} />
        </div>
        <PlanningMap runId={state.currentRunId} result={state.result} />
        <div
          style={{
            padding: 16,
            background: "#fff",
            borderRadius: 12,
            border: "1px solid #e2e8f0",
          }}
        >
          <ResultMetrics
            result={state.result}
            convergence={state.convergence}
            comparison={state.comparison}
            comparisonState={state.comparisonState}
            comparisonError={state.comparisonError}
            onRunComparison={() => {
              if (!state.currentRunId) return Promise.resolve(null);
              return runQuickComparison(state.currentRunId);
            }}
          />
        </div>
        <div
          style={{
            padding: 16,
            background: "#fff",
            borderRadius: 12,
            border: "1px solid #e2e8f0",
          }}
        >
          <ResultDetailPanel result={state.result} />
        </div>
      </main>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState("planning");
  const { state: runState } = useRun();
  const { dispatch } = useMonitoring();

  const goMonitoring = () => {
    if (runState.currentRunId) {
      dispatch({ type: "ARM_MONITOR", payload: runState.currentRunId });
      setTab("monitoring");
    }
  };

  return (
    <div style={{ padding: 16, maxWidth: "100%", boxSizing: "border-box" }}>
      <WebSocketBridge />
      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        <button onClick={() => setTab("planning")}>Planning</button>
        <button onClick={() => setTab("monitoring")}>Monitoring</button>
      </div>
      {tab === "planning" ? <PlanningTab onGoMonitoring={goMonitoring} /> : <MonitoringView />}
    </div>
  );
}

