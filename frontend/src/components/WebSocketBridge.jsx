import { useEffect } from "react";
import { useRun } from "../context/RunContext";
import { useMonitoring } from "../context/MonitoringContext";

export default function WebSocketBridge() {
  const { dispatch: runDispatch } = useRun();
  const { dispatch: monDispatch } = useMonitoring();

  useEffect(() => {
    const api = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
    const wsUrl = api.replace("http://", "ws://").replace("https://", "wss://") + "/ws";
    let stopped = false;
    let socket = null;
    let retry = 0;
    const maxRetries = 5;

    const connect = () => {
      if (stopped) return;
      socket = new WebSocket(wsUrl);
      socket.onopen = () => {
        retry = 0;
      };
      socket.onmessage = (evt) => {
        try {
          const msg = JSON.parse(evt.data);
          const receivedAt = Date.now();
          runDispatch({ type: "WS_MESSAGE", payload: msg });
          monDispatch({ type: "WS_MESSAGE", payload: msg, receivedAt });
        } catch {
          return;
        }
      };
      socket.onclose = () => {
        if (stopped) return;
        if (retry >= maxRetries) {
          // Fallback to HTTP polling handled in contexts; stop WS retries.
          return;
        }
        const waitMs = Math.min(1000 * 2 ** retry, 10000);
        retry += 1;
        setTimeout(connect, waitMs);
      };
    };

    connect();
    return () => {
      stopped = true;
      if (socket) socket.close();
    };
  }, [runDispatch, monDispatch]);

  return null;
}

