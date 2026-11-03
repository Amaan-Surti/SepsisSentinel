import { fmtValue, Variable } from "../variables";

/** One measurement over the stay so far: dots on measured hours joined by a line, onset and "now" marked. */
export function MiniSeries({
  v,
  values,
  hours,
  xMax,
  onset,
}: {
  v: Variable;
  values: (number | null)[];
  hours: number[];
  xMax: number;
  onset: number | null;
}) {
  const pts = hours.flatMap((h, i) => (values[i] == null ? [] : [{ h, y: values[i] as number }]));
  const last = pts[pts.length - 1];
  if (!last) {
    return (
      <div className="mini empty">
        <div className="mini-head">
          <span className="mini-name">{v.label}</span>
          <span className="mini-val">—</span>
        </div>
        <div className="mini-meta">
          <span>not measured yet</span>
          <span>{v.unit}</span>
        </div>
      </div>
    );
  }
  const W = 220;
  const H = 54;
  const lo = Math.min(...pts.map((p) => p.y));
  const hi = Math.max(...pts.map((p) => p.y));
  const pad = hi === lo ? Math.max(Math.abs(hi) * 0.05, 1) : (hi - lo) * 0.12;
  const x = (h: number) => 4 + (h / Math.max(xMax, 1)) * (W - 8);
  const y = (val: number) => H - 4 - ((val - (lo - pad)) / (hi - lo + 2 * pad)) * (H - 8);
  const nowH = hours[hours.length - 1];
  const summary = `${v.label}: ${pts.length} measurement${pts.length === 1 ? "" : "s"}, last ${fmtValue(last.y, v.digits)} ${v.unit} at hour ${last.h}, range ${fmtValue(lo, v.digits)} to ${fmtValue(hi, v.digits)}`;
  return (
    <div className="mini">
      <div className="mini-head">
        <span className="mini-name">{v.label}</span>
        <span className="mini-val">
          {fmtValue(last.y, v.digits)} <span className="unit">{v.unit}</span>
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={summary}>
        {onset != null && onset <= xMax && <line x1={x(onset)} x2={x(onset)} y1={0} y2={H} className="on" />}
        <line x1={x(nowH)} x2={x(nowH)} y1={0} y2={H} className="nw" />
        {pts.length > 1 && <polyline className="ln" points={pts.map((p) => `${x(p.h)},${y(p.y)}`).join(" ")} />}
        {pts.map((p) => (
          <circle key={p.h} className="pt" cx={x(p.h)} cy={y(p.y)} r={2.4} />
        ))}
      </svg>
      <div className="mini-meta">
        <span>
          {pts.length}× · last at hour {last.h}
        </span>
        <span className="range">
          {fmtValue(lo, v.digits)}–{fmtValue(hi, v.digits)}
        </span>
      </div>
    </div>
  );
}
