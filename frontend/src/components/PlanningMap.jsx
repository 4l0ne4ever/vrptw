import { useEffect, useMemo, useState } from "react";
import RouteMap from "./RouteMap";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function straightPolylinesFromRoutes(routes, depot, customers) {
  if (!depot || !Array.isArray(routes)) return [];
  const byId = new Map((customers || []).map((c) => [Number(c.id), c]));
  return routes.map((route, idx) => {
    const coordinates = [];
    for (const node of route || []) {
      const id = Number(node);
      if (id === 0) {
        coordinates.push({ lat: depot.lat, lon: depot.lon });
        continue;
      }
      const c = byId.get(id);
      if (c) coordinates.push({ lat: c.lat, lon: c.lon });
    }
    return { route_id: idx + 1, coordinates };
  });
}

export default function PlanningMap({ runId, result }) {
  const [ctx, setCtx] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    if (!runId || !result) {
      setCtx(null);
      setErr(null);
      return;
    }
    let cancelled = false;
    setErr(null);
    fetch(`${API_URL}/monitor/context?run_id=${encodeURIComponent(runId)}&replay_slot=0`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((d) => {
        if (!cancelled) setCtx(d);
      })
      .catch((e) => {
        if (!cancelled) setErr(e?.message || "Map load failed");
      });
    return () => {
      cancelled = true;
    };
  }, [runId, result]);

  const polylines = useMemo(() => {
    if (!ctx?.depot) return [];
    const fromServer = ctx.polylines || [];
    const hasGeometry = fromServer.some((p) => (p.coordinates || []).length > 1);
    if (hasGeometry) return fromServer;
    const routes = result?.solution?.routes || [];
    return straightPolylinesFromRoutes(routes, ctx.depot, ctx.customers);
  }, [ctx, result]);

  const customerRouteIndex = useMemo(() => {
    const out = {};
    (result?.solution?.routes || []).forEach((route, ri) => {
      for (const n of route || []) {
        const id = Number(n);
        if (id !== 0) out[id] = ri;
      }
    });
    return out;
  }, [result]);

  if (!runId || !result) return null;

  return (
    <section
      style={{
        display: "grid",
        gap: 10,
        minWidth: 0,
        padding: 14,
        background: "#fff",
        border: "1px solid #e2e8f0",
        borderRadius: 12,
        boxShadow: "0 1px 3px rgba(15,23,42,0.06)",
      }}
    >
      <h3 style={{ margin: 0, fontSize: 17, color: "#0f172a" }}>Bản đồ lộ trình</h3>
      <div style={{ fontSize: 13, color: "#64748b", lineHeight: 1.45 }}>
        Đường màu theo từng xe; điểm tròn màu trùng tuyến. Di chuột vào điểm để xem mã khách. Kho: vòng tròn xanh
        có nhãn.
      </div>
      {err ? <div style={{ color: "#b91c1c" }}>{err}</div> : null}
      {!ctx && !err ? <div>Loading map…</div> : null}
      {ctx ? (
        <RouteMap
          depot={ctx.depot}
          customers={ctx.customers}
          polylines={polylines}
          telemetry={{}}
          trails={{}}
          alerts={[]}
          mapHeight={480}
          customerRouteIndex={customerRouteIndex}
          variant="planning"
          depotCompact={false}
        />
      ) : null}
    </section>
  );
}
