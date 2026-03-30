import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export default function ConvergenceChart({ data }) {
  if (!data || data.length === 0) return <div>No data</div>;
  return (
    <div style={{ width: "100%", height: 260 }}>
      <ResponsiveContainer>
        <LineChart data={data}>
          <XAxis dataKey="generation" />
          <YAxis />
          <Tooltip />
          <Line type="monotone" dataKey="best_fitness" stroke="#2563eb" dot={false} />
          <Line type="monotone" dataKey="avg_fitness" stroke="#16a34a" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

