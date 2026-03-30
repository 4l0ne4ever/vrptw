import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { MonitoringProvider } from "./context/MonitoringContext";
import { RunProvider } from "./context/RunContext";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <RunProvider>
      <MonitoringProvider>
        <App />
      </MonitoringProvider>
    </RunProvider>
  </React.StrictMode>
);

