import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { PatientDetail } from "../api";
import { C, TICK } from "../chartColors";

const VITALS: { key: string; label: string; unit: string }[] = [
  { key: "HR", label: "Heart rate", unit: "bpm" },
  { key: "MAP", label: "Mean arterial pressure", unit: "mmHg" },
  { key: "Resp", label: "Respiratory rate", unit: "/min" },
  { key: "Temp", label: "Temperature", unit: "°C" },
  { key: "O2Sat", label: "SpO₂", unit: "%" },
];

function lastValue(xs: (number | null)[]): number | null {
  for (let i = xs.length - 1; i >= 0; i--) if (xs[i] != null) return xs[i];
  return null;
}

export function VitalsGrid({ p, showTruth }: { p: PatientDetail; showTruth: boolean }) {
  const xMax = p.stay_hours - 1;
  return (
    <div className="vitals-grid">
      {VITALS.map((v) => {
        const series = p.vitals[v.key] ?? [];
        const data = p.hours.map((h, i) => ({ hour: h, value: series[i] }));
        const last = lastValue(series);
        return (
          <div className="mini-chart" key={v.key}>
            <div className="mini-head">
              <span>{v.label}</span>
              <span className="num-tab">
                {last == null ? "—" : last.toFixed(v.key === "Temp" ? 1 : 0)} <span className="unit">{v.unit}</span>
              </span>
            </div>
            <ResponsiveContainer width="100%" height={132}>
              <LineChart data={data} margin={{ top: 6, right: 12, bottom: 0, left: 0 }}>
                <CartesianGrid stroke={C.grid} vertical={false} />
                <XAxis dataKey="hour" type="number" domain={[0, xMax]} tickCount={5} allowDecimals={false} stroke={C.muted} fontSize={TICK.mini} height={22} />
                <YAxis domain={["auto", "auto"]} stroke={C.muted} fontSize={TICK.mini} width={42} tickCount={3} />
                {showTruth && p.onset_hour != null && <ReferenceLine x={p.onset_hour} stroke={C.text} strokeWidth={2} />}
                <Tooltip formatter={(x: number) => [`${x} ${v.unit}`, v.label]} labelFormatter={(h) => `Hour ${h}`} />
                <Line dataKey="value" stroke={C.text} strokeWidth={2} dot={{ r: 2 }} connectNulls isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        );
      })}
    </div>
  );
}
