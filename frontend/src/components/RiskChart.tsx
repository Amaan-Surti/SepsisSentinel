import { CartesianGrid, ComposedChart, Line, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { PatientDetail, pct } from "../api";
import { C, TICK } from "../chartColors";

/** Contiguous alert runs [start, end] in hour units. */
export function alertRuns(hours: number[], alert: boolean[]): [number, number][] {
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

/** Round an axis maximum up to a readable value (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8 or 10 × 10^k). */
function niceMax(x: number): number {
  const k = Math.pow(10, Math.floor(Math.log10(x)));
  const m = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10].find((f) => f * k >= x - 1e-9) ?? 10;
  return m * k;
}

/** One-sentence text alternative for the chart. */
export function riskSummary(p: PatientDetail): string {
  const runs = alertRuns(p.hours, p.alert);
  const peak = Math.max(...p.risk);
  const alertTxt = runs.length
    ? `in alert during ${runs.length === 1 && runs[0][0] === runs[0][1] ? "hour" : "hours"} ${runs.map(([a, b]) => (a === b ? `${a}` : `${a}–${b}`)).join(", ")}`
    : "no alert so far";
  return `Risk ${pct(p.risk[p.risk.length - 1], 2)} at hour ${p.hour}, peak ${pct(peak, 2)}; threshold ${pct(p.threshold, 1)}; ${alertTxt}.`;
}

export function RiskChart({ p, showTruth }: { p: PatientDetail; showTruth: boolean }) {
  const data = p.hours.map((h, i) => ({ hour: h, risk: +(p.risk[i] * 100).toFixed(3) }));
  const tau = p.threshold * 100;
  const yMax = niceMax(Math.max(tau * 2, ...data.map((d) => d.risk)) * 1.08);
  const xMax = p.stay_hours - 1;
  return (
    <figure className="chart" role="img" aria-label={riskSummary(p)}>
      <ResponsiveContainer width="100%" height={360}>
        <ComposedChart data={data} margin={{ top: 22, right: 28, bottom: 18, left: 4 }}>
          <CartesianGrid stroke={C.grid} vertical={false} />
          <XAxis
            dataKey="hour"
            type="number"
            domain={[0, xMax]}
            tickCount={10}
            allowDecimals={false}
            stroke={C.muted}
            fontSize={TICK.main}
            label={{ value: "ICU hour", position: "insideBottomRight", offset: -12, fill: C.muted, fontSize: TICK.main }}
          />
          <YAxis domain={[0, yMax]} unit="%" stroke={C.muted} fontSize={TICK.main} width={52} />
          {alertRuns(p.hours, p.alert).map(([a, b]) => (
            <ReferenceArea key={a} x1={a - 0.5} x2={b + 0.5} fill={C.alert} fillOpacity={0.2} ifOverflow="hidden" />
          ))}
          <ReferenceLine
            y={tau}
            stroke={C.alertInk}
            strokeWidth={2}
            strokeDasharray="6 4"
            label={{ value: `Threshold ${tau.toFixed(1)}%`, position: "insideTopLeft", fill: C.alertInk, fontSize: TICK.main, fontWeight: 600 }}
          />
          {p.hour < xMax && <ReferenceLine x={p.hour} stroke={C.muted} strokeDasharray="2 3" label={{ value: "now", position: "top", fill: C.muted, fontSize: TICK.mini }} />}
          {showTruth && p.onset_hour != null && p.onset_hour <= xMax && (
            <ReferenceLine
              x={p.onset_hour}
              stroke={C.text}
              strokeWidth={2.5}
              label={{ value: "Sepsis onset", position: "top", fill: C.text, fontSize: TICK.main, fontWeight: 600 }}
            />
          )}
          <Tooltip formatter={(v: number) => [`${v.toFixed(2)}%`, "Risk"]} labelFormatter={(h) => `ICU hour ${h}`} />
          <Line type="monotone" dataKey="risk" stroke={C.teal} strokeWidth={3} dot={false} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </figure>
  );
}
