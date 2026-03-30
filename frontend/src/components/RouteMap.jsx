import { CircleMarker, MapContainer, Marker, Polyline, TileLayer, Tooltip } from "react-leaflet";
import L from "leaflet";
import iconRetina from "leaflet/dist/images/marker-icon-2x.png";
import iconUrl from "leaflet/dist/images/marker-icon.png";
import shadowUrl from "leaflet/dist/images/marker-shadow.png";
import "leaflet/dist/leaflet.css";

const ROUTE_LINE_COLORS = ["#2563eb", "#16a34a", "#ca8a04", "#9333ea", "#dc2626", "#0891b2"];

function depotIcon(sizePx, fontPx) {
  return L.divIcon({
    className: "vrptw-depot-marker",
    html: `<div style="width:${sizePx}px;height:${sizePx}px;border-radius:50%;background:#1d4ed8;color:#fff;display:flex;align-items:center;justify-content:center;font-size:${fontPx}px;font-weight:700;border:2px solid #fff;box-shadow:0 2px 8px rgba(0,0,0,.35)">Kho</div>`,
    iconSize: [sizePx, sizePx],
    iconAnchor: [sizePx / 2, sizePx / 2],
  });
}

const depotDivIcon = depotIcon(40, 12);
const depotDivIconCompact = depotIcon(28, 10);

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({ iconRetinaUrl: iconRetina, iconUrl, shadowUrl });

function isVisited(visitedCustomerIds, id) {
  const n = Number(id);
  return (visitedCustomerIds || []).some((x) => Number(x) === n);
}

export default function RouteMap({
  depot,
  customers,
  telemetry,
  trails,
  polylines,
  alerts,
  mapHeight = 480,
  customerRouteIndex = null,
  variant = "default",
  visitedCustomerIds = [],
  customerVehicleIndex = null,
  depotCompact = false,
}) {
  const center = depot ? [depot.lat, depot.lon] : [21.0285, 105.8542];
  const violatedVehicles = new Set(
    (alerts || [])
      .filter((a) => String(a?.type || "") === "tw_violation" && a?.vehicle_id != null)
      .map((a) => String(a.vehicle_id))
  );

  const replayMode = variant === "replay";
  const plannedPolylines = (polylines || []).filter((p) => (p.coordinates || []).length >= 2);

  return (
    <MapContainer center={center} zoom={12} style={{ width: "100%", height: mapHeight }}>
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {plannedPolylines.map((p) => {
        const coords = (p.coordinates || []).map((x) => [x.lat, x.lon]).filter((pt) => pt[0] != null && pt[1] != null);
        if (coords.length < 2) return null;
        const color = ROUTE_LINE_COLORS[(Number(p.route_id) - 1) % ROUTE_LINE_COLORS.length];
        const pathOptions = replayMode
          ? { color, weight: 4, opacity: 0.3 }
          : { color, weight: 5, opacity: 0.88 };
        return <Polyline key={`plan-${p.route_id}`} positions={coords} pathOptions={pathOptions} />;
      })}
      {Object.entries(trails || {}).map(([k, trail]) => {
        if (!trail || trail.length < 2) return null;
        const vid = Number(k);
        const color = ROUTE_LINE_COLORS[(vid - 1) % ROUTE_LINE_COLORS.length];
        return <Polyline key={`trail-${k}`} positions={trail} pathOptions={{ color, weight: 5, opacity: 0.92 }} />;
      })}
      {depot ? (
        <Marker position={[depot.lat, depot.lon]} icon={depotCompact ? depotDivIconCompact : depotDivIcon} />
      ) : null}
      {(customers || []).map((c) => {
        const ri =
          customerRouteIndex != null
            ? customerRouteIndex[c.id] ?? customerRouteIndex[Number(c.id)]
            : customerVehicleIndex != null
              ? customerVehicleIndex[String(c.id)] ?? customerVehicleIndex[c.id]
              : undefined;
        const visited = isVisited(visitedCustomerIds, c.id);
        let fill = "#0f7668";
        if (ri != null && ri !== undefined) {
          fill = ROUTE_LINE_COLORS[Number(ri) % ROUTE_LINE_COLORS.length];
        }
        if (replayMode && !visited) {
          fill = "#cbd5e1";
        }
        return (
          <CircleMarker
            key={c.id}
            center={[c.lat, c.lon]}
            radius={12}
            pathOptions={{
              color: "#ffffff",
              weight: 3,
              fillColor: fill,
              fillOpacity: replayMode && !visited ? 0.75 : 0.95,
            }}
          >
            <Tooltip direction="top" offset={[0, -10]} opacity={0.95}>
              {ri != null ? `Điểm ${c.id} · tuyến ${Number(ri) + 1}` : `Điểm ${c.id}`}
              {replayMode && !visited ? " · chưa ghé" : visited ? " · đã ghé" : ""}
              {c.demand != null ? ` · nhu cầu ${c.demand}` : ""}
            </Tooltip>
          </CircleMarker>
        );
      })}
      {Object.entries(telemetry || {}).map(([k, t]) => (
        <CircleMarker
          key={`veh-${k}`}
          center={[t.lat, t.lon]}
          radius={10}
          pathOptions={{
            color: "#fff",
            weight: 3,
            fillColor: violatedVehicles.has(String(t.vehicle_id)) ? "#dc2626" : ROUTE_LINE_COLORS[(Number(t.vehicle_id) - 1) % ROUTE_LINE_COLORS.length],
            fillOpacity: 0.95,
          }}
        >
          <Tooltip permanent direction="top" offset={[0, -10]} opacity={0.95}>
            Xe {t.vehicle_id} · {t.status}
            {t.speed_kmh != null ? ` · ${Number(t.speed_kmh).toFixed(1)} km/h` : ""}
            {t.traffic_factor != null ? ` · tắc ×${Number(t.traffic_factor).toFixed(2)}` : ""}
          </Tooltip>
        </CircleMarker>
      ))}
    </MapContainer>
  );
}
