export default function AlertFeed({ alerts }) {
  const rows = alerts || [];
  const shown = rows.slice(0, 50);
  const colorFor = (a) => {
    const t = String(a?.type || "");
    if (t === "tw_violation") return "#dc2626";
    return "#f59e0b";
  };
  return (
    <div style={{ maxHeight: 220, overflow: "auto" }}>
      {shown.length === 0 ? <div>No alerts</div> : null}
      {shown.map((a, idx) => (
        <div key={idx} style={{ color: colorFor(a), fontSize: 12, marginBottom: 2 }}>
          [{a.type}] vehicle={a.vehicle_id} customer={a.customer_id} late={a.lateness_minutes}
        </div>
      ))}
    </div>
  );
}

