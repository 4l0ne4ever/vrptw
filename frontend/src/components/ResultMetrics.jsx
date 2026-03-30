function formatKm(v) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  return `${Number(v).toLocaleString("vi-VN", { maximumFractionDigits: 2 })} km`;
}

function formatMoney(v) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  const n = Number(v);
  return `${n.toLocaleString("vi-VN", { maximumFractionDigits: 0 })}`;
}

function formatPlain(v, digits = 2) {
  if (v == null || Number.isNaN(Number(v))) return "—";
  const n = Number(v);
  if (Math.abs(n) >= 1e9) return n.toLocaleString("vi-VN", { maximumFractionDigits: 2, notation: "compact" });
  return n.toLocaleString("vi-VN", { minimumFractionDigits: 0, maximumFractionDigits: digits });
}

function formatBool(v) {
  if (v === true) return "Dat";
  if (v === false) return "Khong";
  return "—";
}

function Row({ label, value }) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "minmax(0, 1fr) auto",
        gap: 12,
        padding: "6px 0",
        borderBottom: "1px solid #e8e8e8",
        fontSize: 13,
      }}
    >
      <span style={{ color: "#525252" }}>{label}</span>
      <span style={{ fontVariantNumeric: "tabular-nums", textAlign: "right", fontWeight: 500 }}>{value}</span>
    </div>
  );
}

function BigCard({ label, value, hint }) {
  return (
    <div
      style={{
        background: "linear-gradient(145deg, #f8fafc 0%, #f1f5f9 100%)",
        border: "1px solid #e2e8f0",
        borderRadius: 10,
        padding: "14px 16px",
        boxShadow: "0 1px 2px rgba(15,23,42,0.06)",
        minWidth: 0,
      }}
    >
      <div style={{ fontSize: 12, color: "#64748b", marginBottom: 6, letterSpacing: "0.02em" }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 700, color: "#0f172a", lineHeight: 1.2 }}>{value}</div>
      {hint ? (
        <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 8, lineHeight: 1.35 }}>{hint}</div>
      ) : null}
    </div>
  );
}

function DetailsBlock({ title, children, defaultOpen = false }) {
  return (
    <details
      open={defaultOpen}
      style={{
        border: "1px solid #e2e8f0",
        borderRadius: 10,
        background: "#fff",
        overflow: "hidden",
      }}
    >
      <summary
        style={{
          cursor: "pointer",
          padding: "12px 14px",
          fontWeight: 600,
          fontSize: 14,
          color: "#334155",
          listStyle: "none",
          background: "#f8fafc",
        }}
      >
        {title}
      </summary>
      <div style={{ padding: "8px 14px 14px" }}>{children}</div>
    </details>
  );
}

function CompareTable({ rows }) {
  if (!rows?.length) return <div style={{ fontSize: 12, color: "#64748b" }}>Chua co du lieu.</div>;
  return (
    <div style={{ overflowX: "auto", border: "1px solid #e2e8f0", borderRadius: 8 }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
        <thead>
          <tr style={{ background: "#f8fafc", color: "#475569" }}>
            <th style={{ textAlign: "left", padding: "8px 10px", borderBottom: "1px solid #e2e8f0" }}>Phuong an</th>
            <th style={{ textAlign: "right", padding: "8px 10px", borderBottom: "1px solid #e2e8f0" }}>Km</th>
            <th style={{ textAlign: "right", padding: "8px 10px", borderBottom: "1px solid #e2e8f0" }}>Chi phi</th>
            <th style={{ textAlign: "right", padding: "8px 10px", borderBottom: "1px solid #e2e8f0" }}>Xe</th>
            <th style={{ textAlign: "center", padding: "8px 10px", borderBottom: "1px solid #e2e8f0" }}>Feasible</th>
            <th style={{ textAlign: "right", padding: "8px 10px", borderBottom: "1px solid #e2e8f0" }}>TW vio</th>
            <th style={{ textAlign: "right", padding: "8px 10px", borderBottom: "1px solid #e2e8f0" }}>Runtime</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, idx) => (
            <tr key={`${r.label}-${idx}`} style={{ borderBottom: "1px solid #f1f5f9" }}>
              <td style={{ padding: "8px 10px" }}>{r.label}</td>
              <td style={{ padding: "8px 10px", textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{formatPlain(r.total_distance, 2)}</td>
              <td style={{ padding: "8px 10px", textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{formatMoney(r.total_cost)}</td>
              <td style={{ padding: "8px 10px", textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{formatPlain(r.num_routes, 0)}</td>
              <td style={{ padding: "8px 10px", textAlign: "center" }}>{formatBool(r.is_feasible)}</td>
              <td style={{ padding: "8px 10px", textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{formatPlain(r.tw_violations, 0)}</td>
              <td style={{ padding: "8px 10px", textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{`${formatPlain(r.runtime_s, 2)}s`}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function ResultMetrics({
  result,
  convergence,
  comparison = null,
  comparisonState = "idle",
  comparisonError = null,
  onRunComparison = null,
}) {
  const k = result?.kpis;
  if (!k) return <div style={{ color: "#64748b" }}>Chưa có số liệu — chạy xong sẽ hiển thị tại đây.</div>;

  const cv = k.constraint_violations || {};
  const lastGen = convergence?.length ? convergence[convergence.length - 1] : null;
  const feasible = Boolean(k.is_feasible);
  const boolLabel = (v) => (v === true ? "có vấn đề" : v === false ? "ổn" : "—");

  const runLabel = [result?.dataset, result?.dataset_type].filter(Boolean).join(" · ");

  return (
    <div style={{ display: "grid", gap: 16 }}>
      <div>
        <h3 style={{ margin: "0 0 4px", fontSize: 18, color: "#0f172a" }}>Kết quả giao hàng</h3>
        {runLabel ? (
          <div style={{ fontSize: 13, color: "#64748b" }}>{runLabel}</div>
        ) : null}
        {result?.run_id ? (
          <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 4, wordBreak: "break-all" }}>
            Mã run: {result.run_id}
          </div>
        ) : null}
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
          gap: 12,
        }}
      >
        <BigCard
          label="Tổng quãng đường"
          value={formatKm(k.total_distance)}
          hint="Tổng km các xe đi (theo ma trận khoảng cách đã tính)."
        />
        <BigCard
          label="Số xe dùng"
          value={String(k.num_routes ?? "—")}
          hint="Một xe = một tuyến khép kín depot → khách → depot."
        />
        <BigCard
          label="Số điểm phục vụ"
          value={String(k.num_customers ?? "—")}
          hint="Tổng khách được ghé trong lời giải."
        />
        <BigCard
          label="Ràng buộc thời gian"
          value={feasible ? "Đạt" : "Cần xem lại"}
          hint={
            feasible
              ? "Theo kiểm tra nhanh: không phát hiện vi phạm cố định rõ ràng."
              : "Có tín hiệu phạt hoặc vi phạm — xem bảng chi tiết bên dưới."
          }
        />
      </div>

      <p style={{ margin: 0, fontSize: 13, color: "#475569", lineHeight: 1.5 }}>
        Các chỉ số dạng điểm (fitness, hiệu quả nội bộ) chỉ phục vụ so sánh thuật toán; có thể bỏ qua nếu bạn chỉ
        quan tâm km và tuyến trên bản đồ.
      </p>

      <DetailsBlock title="Chi phí & thời gian (ước tính mô hình)" defaultOpen>
        <Row label="Tổng chi phí (đơn vị nội bộ)" value={formatMoney(k.total_cost)} />
        <Row label="Chi phí giao / vận chuyển (shipping)" value={formatMoney(k.shipping_total_cost)} />
        <Row label="Thời gian vận hành ước tính" value={`${formatPlain(k.total_duration_hours, 2)} h`} />
        <Row label="Chi phí trên mỗi km" value={formatPlain(k.cost_per_km, 2)} />
        <Row label="Chi phí trên mỗi điểm" value={formatMoney(k.cost_per_customer)} />
        <Row label="Loại dịch vụ (mô hình)" value={String(k.shipping_service_type ?? "—")} />
      </DetailsBlock>

      <DetailsBlock title="Chi tiết kỹ thuật (GA & tải trọng)">
        {lastGen ? (
          <Row label="Số thế hệ đã ghi" value={`${convergence.length} (đến ${lastGen.generation})`} />
        ) : null}
        <Row label="Fitness (càng cao càng tốt trong GA)" value={formatPlain(k.fitness, 4)} />
        <Row label="Phạt (penalty)" value={formatPlain(k.penalty, 4)} />
        <Row label="Điểm chất lượng nội bộ" value={formatPlain(k.solution_quality, 4)} />
        <Row label="Điểm hiệu quả nội bộ" value={formatPlain(k.efficiency_score, 4)} />
        <Row label="Điểm khả thi nội bộ" value={formatPlain(k.feasibility_score, 4)} />
        <Row label="Điểm dừng trung bình / tuyến" value={formatPlain(k.avg_route_length, 2)} />
        <Row label="Tải trung bình (% capacity)" value={`${formatPlain(k.avg_utilization, 1)} %`} />
        <Row label="Tải max / min (%)" value={`${formatPlain(k.max_utilization, 1)} / ${formatPlain(k.min_utilization, 1)}`} />
      </DetailsBlock>

      <DetailsBlock title="Ràng buộc (kiểm tra nhanh)">
        <Row label="Tín hiệu tổng hợp" value={String(cv.total_violations ?? "—")} />
        <Row label="Tải xe" value={boolLabel(cv.capacity_violations)} />
        <Row label="Số xe" value={boolLabel(cv.vehicle_count_violations)} />
        <Row label="Thăm khách" value={boolLabel(cv.customer_visit_violations)} />
        <Row label="Depot" value={boolLabel(cv.depot_violations)} />
        <Row label="Sự kiện cửa sổ thời gian" value={String(cv.time_window_violations ?? "—")} />
      </DetailsBlock>

      <DetailsBlock title="So sanh nhanh (beta)">
        <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 10, flexWrap: "wrap" }}>
          <button
            type="button"
            disabled={!onRunComparison || comparisonState === "running"}
            onClick={() => onRunComparison?.().catch(() => undefined)}
          >
            {comparisonState === "running" ? "Dang chay so sanh..." : "Chay HGA vs GA vs NN + traffic"}
          </button>
          {comparisonError ? <span style={{ color: "#b91c1c", fontSize: 12 }}>{comparisonError}</span> : null}
        </div>
        <div style={{ fontSize: 12, color: "#64748b", marginBottom: 10 }}>
          Cac bang duoi day la benchmark nhanh de so sanh tuong doi, khong thay the ket qua production full-run.
        </div>

        <div style={{ display: "grid", gap: 12 }}>
          <div>
            <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 6 }}>HGA vs GA thuong vs NN</div>
            <CompareTable rows={comparison?.algorithm_compare || []} />
          </div>
          <div>
            <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 6 }}>Co traffic dong vs khong traffic dong (HGA)</div>
            <CompareTable rows={comparison?.traffic_compare || []} />
          </div>
          <div>
            <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 6 }}>Co re-plan vs khong re-plan</div>
            <CompareTable rows={comparison?.replan_compare?.rows || []} />
          </div>
          <div>
            <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 6 }}>Monitoring system vs toi uu tinh</div>
            <CompareTable rows={comparison?.monitoring_compare?.rows || []} />
          </div>
          <div style={{ border: "1px solid #e2e8f0", borderRadius: 8, padding: 10, fontSize: 12, color: "#475569" }}>
            <div>
              Re-plan:{" "}
              {comparison?.replan_compare
                ? comparison.replan_compare.has_replan
                  ? `Da co (revision ${comparison.replan_compare.plan_revision})`
                  : "Chua co"
                : "—"}
            </div>
            <div style={{ marginTop: 4 }}>
              Replan running: {comparison?.replan_compare?.replan_running ? "Dang chay" : "Khong"} |{" "}
              Monitoring system: {comparison?.monitoring_compare?.supports_monitoring ? "Co" : "—"} | Toi uu tinh:{" "}
              {comparison?.monitoring_compare?.static_optimization ? "Co" : "—"}
            </div>
          </div>
        </div>
      </DetailsBlock>
    </div>
  );
}
