const ROUTE_ACCENT = ["#2563eb", "#16a34a", "#ca8a04", "#9333ea", "#dc2626", "#0891b2"];

/** Decimal hours (e.g. 8.0833) → HH:mm same day */
function toClock(h) {
  if (h == null || Number.isNaN(Number(h))) return "—";
  const totalMin = Math.round(Number(h) * 60);
  const hh = Math.floor(totalMin / 60) % 48;
  const mm = ((totalMin % 60) + 60) % 60;
  return `${String(hh).padStart(2, "0")}:${String(mm).padStart(2, "0")}`;
}

function countStops(route) {
  if (!route?.length) return 0;
  return route.filter((id) => Number(id) !== 0).length;
}

function twSummary(stops) {
  if (!stops?.length) return "—";
  let bad = 0;
  for (const s of stops) {
    if (s?.violated) bad += 1;
  }
  return bad === 0 ? "Ổn" : `${bad} điểm có vấn đề`;
}

function routeViolations(stops) {
  if (!stops?.length) return 0;
  return stops.filter((s) => s?.violated).length;
}

export default function ResultDetailPanel({ result }) {
  const routes = result?.solution?.routes || [];
  const details = result?.solution?.time_window_details || [];

  if (routes.length === 0) {
    return <div style={{ color: "#64748b", fontSize: 14 }}>Chưa có tuyến trong kết quả.</div>;
  }

  return (
    <section style={{ display: "grid", gap: 20, minWidth: 0, fontFamily: "'Inter', system-ui, sans-serif" }}>
      <div>
        <h3 style={{ margin: "0 0 4px", fontSize: 17, fontWeight: 600, color: "#0f172a", letterSpacing: "-0.02em" }}>
          Chi tiết theo xe
        </h3>
        <p style={{ margin: 0, fontSize: 13, color: "#64748b" }}>Tóm tắt tuyến và đồng hồ đến / bắt đầu phục vụ (giờ trong ngày).</p>
      </div>

      <div
        style={{
          borderRadius: 10,
          border: "1px solid #e2e8f0",
          overflow: "hidden",
          boxShadow: "0 1px 2px rgba(15,23,42,0.04)",
        }}
      >
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <thead>
            <tr style={{ background: "#f8fafc", borderBottom: "1px solid #e2e8f0" }}>
              <th style={{ padding: "12px 14px", textAlign: "left", fontWeight: 600, color: "#475569", width: 72 }}>
                Xe
              </th>
              <th style={{ padding: "12px 14px", textAlign: "left", fontWeight: 600, color: "#475569", width: 88 }}>
                Điểm
              </th>
              <th style={{ padding: "12px 14px", textAlign: "left", fontWeight: 600, color: "#475569", width: 120 }}>
                Khung giờ
              </th>
              <th style={{ padding: "12px 14px", textAlign: "left", fontWeight: 600, color: "#475569" }}>
                Lộ trình
              </th>
            </tr>
          </thead>
          <tbody>
            {routes.map((r, idx) => {
              const d = details[idx];
              const accent = ROUTE_ACCENT[idx % ROUTE_ACCENT.length];
              const seq = (r || []).join(" → ");
              const ok = twSummary(d?.stops) === "Ổn";
              return (
                <tr key={idx} style={{ borderBottom: "1px solid #f1f5f9", background: idx % 2 === 0 ? "#fff" : "#fafbfc" }}>
                  <td style={{ padding: "12px 14px", verticalAlign: "top" }}>
                    <span
                      style={{
                        display: "inline-flex",
                        alignItems: "center",
                        justifyContent: "center",
                        minWidth: 28,
                        height: 28,
                        borderRadius: 8,
                        background: `${accent}18`,
                        color: accent,
                        fontWeight: 700,
                        fontSize: 13,
                      }}
                    >
                      {idx + 1}
                    </span>
                  </td>
                  <td style={{ padding: "12px 14px", verticalAlign: "top", color: "#334155" }}>{countStops(r)}</td>
                  <td style={{ padding: "12px 14px", verticalAlign: "top" }}>
                    <span
                      style={{
                        display: "inline-block",
                        padding: "2px 10px",
                        borderRadius: 999,
                        fontSize: 12,
                        fontWeight: 500,
                        background: ok ? "#dcfce7" : "#fee2e2",
                        color: ok ? "#166534" : "#b91c1c",
                      }}
                    >
                      {twSummary(d?.stops)}
                    </span>
                  </td>
                  <td
                    style={{
                      padding: "12px 14px",
                      verticalAlign: "top",
                      fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
                      fontSize: 12,
                      color: "#475569",
                      lineHeight: 1.5,
                      wordBreak: "break-word",
                    }}
                  >
                    {seq}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {routes.map((r, idx) => {
        const d = details[idx];
        if (!d?.stops?.length) return null;
        const accent = ROUTE_ACCENT[idx % ROUTE_ACCENT.length];
        const bad = routeViolations(d.stops);
        const showNoteCol = bad > 0;

        return (
          <div
            key={`detail-${idx}`}
            style={{
              borderRadius: 12,
              overflow: "hidden",
              border: "1px solid #e2e8f0",
              background: "#fff",
              boxShadow: "0 2px 8px rgba(15,23,42,0.06)",
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 12,
                padding: "14px 16px",
                background: `linear-gradient(90deg, ${accent}12 0%, #fff 48%)`,
                borderBottom: "1px solid #f1f5f9",
              }}
            >
              <span
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  minWidth: 36,
                  height: 36,
                  borderRadius: 10,
                  background: accent,
                  color: "#fff",
                  fontWeight: 700,
                  fontSize: 15,
                }}
              >
                {idx + 1}
              </span>
              <div>
                <div style={{ fontWeight: 600, fontSize: 15, color: "#0f172a" }}>Lịch trình điểm</div>
                <div style={{ fontSize: 12, color: "#64748b", marginTop: 2 }}>
                  {d.stops.length} điểm
                  {bad === 0 ? " · toàn bộ trong khung giờ" : ` · ${bad} điểm cần xem lại`}
                </div>
              </div>
            </div>

            <div style={{ padding: "4px 0 8px" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr style={{ color: "#64748b", fontSize: 14 }}>
                    <th style={{ textAlign: "left", padding: "10px 16px", fontWeight: 600, width: 88 }}>Khách</th>
                    <th style={{ textAlign: "left", padding: "10px 16px", fontWeight: 600, width: 96 }}>
                      Đến
                    </th>
                    <th style={{ textAlign: "left", padding: "10px 16px", fontWeight: 600, minWidth: 120 }}>
                      Bắt đầu phục vụ
                    </th>
                    {showNoteCol ? (
                      <th style={{ textAlign: "left", padding: "10px 16px", fontWeight: 600 }}>Ghi chú</th>
                    ) : null}
                  </tr>
                </thead>
                <tbody>
                  {d.stops.map((s, i) => (
                    <tr
                      key={i}
                      style={{
                        borderTop: "1px solid #f1f5f9",
                        background: i % 2 === 0 ? "#fafbfc" : "#fff",
                      }}
                    >
                      <td style={{ padding: "10px 16px", fontWeight: 600, color: "#475569" }}>KH {s.customer_id}</td>
                      <td
                        style={{
                          padding: "10px 16px",
                          fontVariantNumeric: "tabular-nums",
                          fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
                          color: "#0f172a",
                        }}
                      >
                        {toClock(s.arrival_h)}
                      </td>
                      <td
                        style={{
                          padding: "10px 16px",
                          fontVariantNumeric: "tabular-nums",
                          fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
                          color: "#0f172a",
                        }}
                      >
                        {toClock(s.start_service_h)}
                      </td>
                      {showNoteCol ? (
                        <td style={{ padding: "10px 16px", fontSize: 12, color: s.violated ? "#b91c1c" : "#64748b" }}>
                          {s.violated ? `Trễ / ngoài cửa sổ · ${Math.round(Number(s.lateness_minutes || 0))} phút` : "—"}
                        </td>
                      ) : null}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        );
      })}
    </section>
  );
}
