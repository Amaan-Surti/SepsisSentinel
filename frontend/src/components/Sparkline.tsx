import { pct } from "../api";

/** 12-hour risk sparkline (inline SVG), with the threshold as a dashed line. */
export function Sparkline({ values, threshold, width = 120, height = 32 }: { values: number[]; threshold: number; width?: number; height?: number }) {
  const max = Math.max(threshold * 1.5, ...values, 1e-6);
  const n = Math.max(values.length, 2);
  const x = (i: number) => (i / (n - 1)) * (width - 6) + 3;
  const y = (v: number) => height - 3 - (v / max) * (height - 6);
  const first = values[0] ?? 0;
  const last = values[values.length - 1] ?? 0;
  return (
    <svg
      className="spark"
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={`Risk over the last ${values.length} hours: ${pct(first, 2)} to ${pct(last, 2)}`}
    >
      <line x1={3} x2={width - 3} y1={y(threshold)} y2={y(threshold)} className="thr" />
      {values.length > 1 && <polyline points={values.map((v, i) => `${x(i)},${y(v)}`).join(" ")} className="line" />}
      <circle cx={x(values.length - 1)} cy={y(last)} r={3.2} className={last >= threshold ? "dot alert" : "dot"} />
    </svg>
  );
}
