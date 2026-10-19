/** 12-hour risk sparkline (inline SVG), with the threshold as a dashed line. */
export function Sparkline({ values, threshold, width = 96, height = 26 }: { values: number[]; threshold: number; width?: number; height?: number }) {
  const max = Math.max(threshold * 1.5, ...values, 1e-6);
  const n = Math.max(values.length, 2);
  const x = (i: number) => (i / (n - 1)) * (width - 4) + 2;
  const y = (v: number) => height - 2 - (v / max) * (height - 4);
  const pts = values.map((v, i) => `${x(i)},${y(v)}`).join(" ");
  const last = values[values.length - 1] ?? 0;
  return (
    <svg className="sparkline" width={width} height={height} role="img" aria-label={`Risk trend, last ${values.length} hours`}>
      <line x1={2} x2={width - 2} y1={y(threshold)} y2={y(threshold)} className="spark-threshold" />
      {values.length > 1 && <polyline points={pts} className="spark-line" />}
      <circle cx={x(values.length - 1)} cy={y(last)} r={2.2} className={last >= threshold ? "spark-dot-alert" : "spark-dot"} />
    </svg>
  );
}
