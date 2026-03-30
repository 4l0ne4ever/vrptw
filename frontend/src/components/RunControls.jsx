import { useState } from "react";
import UploadForm from "./UploadForm";

export default function RunControls({
  instances,
  busy,
  onRun,
  onUpload,
  onRefreshInstances,
  runState,
  onGoMonitoring,
  trafficModel,
  onTrafficModelChange,
  err,
}) {
  const [dataset, setDataset] = useState("");
  const [datasetType, setDatasetType] = useState("test");
  const [preset, setPreset] = useState("standard");
  const [populationSize, setPopulationSize] = useState(100);
  const [generations, setGenerations] = useState(500);
  const [seed, setSeed] = useState(42);
  const [timeLimit, setTimeLimit] = useState(0);
  const [uploadMeta, setUploadMeta] = useState(null);

  const applyPreset = (value) => {
    setPreset(value);
    if (value === "fast") {
      setPopulationSize(60);
      setGenerations(200);
      return;
    }
    if (value === "full") {
      setPopulationSize(150);
      setGenerations(1200);
      return;
    }
    if (value === "benchmark") {
      setPopulationSize(100);
      setGenerations(1000);
      return;
    }
  };

  return (
    <div style={{ display: "grid", gap: 10, minWidth: 0 }}>
      <button onClick={onRefreshInstances}>Refresh Instances</button>
      <select
        value={dataset}
        onChange={(e) => {
          const picked = e.target.value;
          setDataset(picked);
          const item = instances.find((it) => it.dataset === picked);
          if (item?.dataset_type) setDatasetType(item.dataset_type);
        }}
      >
        <option value="">Select dataset</option>
        {instances.map((it) => (
          <option key={`${it.dataset_type}-${it.key}`} value={it.dataset}>
            {it.dataset_type} / {it.dataset}
          </option>
        ))}
      </select>
      <select value={datasetType} onChange={(e) => setDatasetType(e.target.value)}>
        <option value="test">test</option>
        <option value="upload">upload</option>
      </select>
      <select value={preset} onChange={(e) => applyPreset(e.target.value)}>
        <option value="fast">fast</option>
        <option value="standard">standard</option>
        <option value="benchmark">benchmark</option>
        <option value="full">full</option>
      </select>
      <label>
        Population
        <input
          type="number"
          min={2}
          value={populationSize}
          onChange={(e) => setPopulationSize(Number(e.target.value))}
        />
      </label>
      <label>
        Generations
        <input type="number" min={1} value={generations} onChange={(e) => setGenerations(Number(e.target.value))} />
      </label>
      <label>
        Seed
        <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
      </label>
      <label>
        Time limit (s, optional)
        <input type="number" min={0} value={timeLimit} onChange={(e) => setTimeLimit(Number(e.target.value))} />
      </label>
      <label>
        Traffic model
        <select value={trafficModel} onChange={(e) => onTrafficModelChange?.(e.target.value)}>
          <option value="instance_factor">instance_factor</option>
          <option value="adaptive">adaptive</option>
          <option value="manual_override">manual_override</option>
        </select>
      </label>
      <UploadForm
        onUpload={async (file) => {
          const meta = await onUpload(file);
          setUploadMeta(meta);
          if (meta?.dataset_id) {
            setDataset(meta.dataset_id);
            setDatasetType("upload");
          }
        }}
      />
      {uploadMeta?.dataset_id ? <div>Uploaded dataset_id: {uploadMeta.dataset_id}</div> : null}
      <button
        disabled={busy || !dataset}
        onClick={() =>
          onRun({
            dataset,
            dataset_type: datasetType,
            population_size: populationSize,
            generations,
            seed,
            time_limit: timeLimit > 0 ? timeLimit : undefined,
            traffic_model: trafficModel,
          })
        }
      >
        Run
      </button>
      {err ? <div style={{ color: "#b91c1c" }}>{err}</div> : null}
      {runState === "complete" ? <button onClick={onGoMonitoring}>Go to Monitoring</button> : null}
    </div>
  );
}

