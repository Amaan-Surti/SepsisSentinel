import { CartesianGrid, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import { PatientDetail } from "../api";
import { C } from "./RiskChart";

const LABS: { key: string; label: string; unit: string }[] = [
  { key: "Lactate", label: "Lactate", unit: "mmol/L" },
  { key: "WBC", label: "White cell count", unit: "×10³/µL" },
  { key: "Creatinine", label: "Creatinine", unit: "mg/dL" },
  { key: "Platelets", label: "Platelets", unit: "×10³/µL" },
];

/** Dot plots: labs are sparse, so only measured hours are drawn. */
export function LabsChart({ p, showTruth }: { p: PatientDetail; showTruth: boolean }) {
  const xMax = p.stay_hours - 1;
  return (
    <div className="labs-grid">
      {LABS.map((l) => {
        const series = p.labs[l.key] ?? [];
        const pts = p.hours.flatMap((h, i) => (series[i] == null ? [] : [{ hour: h, value: series[i] as number }]));
        const last = pts[pts.length - 1];
        return (
          <div className="mini-chart" key={l.key}>
            <div className="mini-head">
              <span>{l.label}</span>
              <span className="mono">
                {last ? last.value.toFixed(1) : "not measured"} {last && <small>{l.unit} · h{last.hour}</small>}
              </span>
            </div>
            <ResponsiveContainer width="100%" height={92}>
              <ScatterChart margin={{ top: 6, right: 12, bottom: 0, left: 0 }}>
                <CartesianGrid stroke={C.grid} vertical={false} />
                <XAxis dataKey="hour" type="number" domain={[0, xMax]} hide />
                <YAxis dataKey="value" type="number" domain={["auto", "auto"]} stroke={C.muted} fontSize={10} width={36} />
                {showTruth && p.onset_hour != null && <ReferenceLine x={p.onset_hour} stroke={C.text} strokeWidth={1} />}
                <Tooltip formatter={(x: number, name: string) => (name === "hour" ? [x, "Hour"] : [`${x} ${l.unit}`, l.label])} />
                <Scatter data={pts} fill={C.teal} isAnimationActive={false} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        );
      })}
    </div>
  );
}
