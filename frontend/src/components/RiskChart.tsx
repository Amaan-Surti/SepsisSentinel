import { CartesianGrid, ComposedChart, Line, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { PatientDetail } from "../api";

export const C = {
  text: "#16303A",
  muted: "#5B6F76",
  grid: "#DDE3DF",
  teal: "#0E7C7B",
  alert: "#D9622B",
  watch: "#B7791F",
};

/** Contiguous alert runs [start, end] in hour units. */
function alertRuns(hours: number[], alert: boolean[]): [number, number][] {
  const runs: [number, number][] = [];
  let start: number | null = null;
  alert.forEach((a, i) => {
    if (a && start === null) start = hours[i];
    if (!a && start !== null) {
      runs.push([start, hours[i - 1]]);
      start = null;
    }
  });
  if (start !== null) runs.push([start, hours[hours.length - 1]]);
  return runs;
}

export function RiskChart({ p, showTruth }: { p: PatientDetail; showTruth: boolean }) {
  const data = p.hours.map((h, i) => ({ hour: h, risk: +(p.risk[i] * 100).toFixed(3) }));
  const tau = p.threshold * 100;
  const yMax = Math.max(tau * 2, ...data.map((d) => d.risk)) * 1.1;
  const xMax = p.stay_hours - 1;
  return (
    <ResponsiveContainer width="100%" height={260}>
      <ComposedChart data={data} margin={{ top: 18, right: 24, bottom: 4, left: 0 }}>
        <CartesianGrid stroke={C.grid} vertical={false} />
        <XAxis dataKey="hour" type="number" domain={[0, xMax]} tickCount={10} allowDecimals={false} stroke={C.muted} fontSize={12} />
        <YAxis domain={[0, +yMax.toFixed(1)]} unit="%" stroke={C.muted} fontSize={12} width={48} />
        {alertRuns(p.hours, p.alert).map(([a, b]) => (
          <ReferenceArea key={a} x1={a - 0.5} x2={b + 0.5} fill={C.alert} fillOpacity={0.13} ifOverflow="hidden" />
        ))}
        <ReferenceLine y={tau} stroke={C.alert} strokeDasharray="5 4" label={{ value: `Threshold ${tau.toFixed(1)}%`, position: "insideTopRight", fill: C.alert, fontSize: 11 }} />
        {showTruth && p.onset_hour != null && p.onset_hour <= xMax && (
          <ReferenceLine x={p.onset_hour} stroke={C.text} strokeWidth={1.5} label={{ value: "Sepsis onset", position: "top", fill: C.text, fontSize: 11 }} />
        )}
        <Tooltip formatter={(v: number) => [`${v.toFixed(2)}%`, "Risk"]} labelFormatter={(h) => `ICU hour ${h}`} />
        <Line type="monotone" dataKey="risk" stroke={C.teal} strokeWidth={2.2} dot={false} isAnimationActive={false} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
