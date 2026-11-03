import { useRef, useState } from "react";
import { fmtNormal, fmtValue, Variable, whereVsNormal } from "../variables";

const WHERE = { low: "below normal", normal: "in normal range", high: "above normal" } as const;

/** One measurement over the stay so far: dots on measured hours joined by a line, the normal range shaded,
 * onset and "now" marked. Hovering shows a box with the nearest measured hour's value. */
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
  const svgRef = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  const pts = hours.flatMap((h, i) => (values[i] == null ? [] : [{ h, y: values[i] as number }]));
  const last = pts[pts.length - 1];
  const normal = <span className="mini-normal">normal {fmtNormal(v)}</span>;
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
        <div className="mini-meta">{normal}</div>
      </div>
    );
  }
  const W = 220;
  const H = 54;
  // The y scale always includes the normal range, so you can see where the patient sits against it.
  const lo = Math.min(v.normal[0], ...pts.map((p) => p.y));
  const hi = Math.max(v.normal[1], ...pts.map((p) => p.y));
  const pad = hi === lo ? Math.max(Math.abs(hi) * 0.05, 1) : (hi - lo) * 0.12;
  const x = (h: number) => 4 + (h / Math.max(xMax, 1)) * (W - 8);
  const y = (val: number) => H - 4 - ((val - (lo - pad)) / (hi - lo + 2 * pad)) * (H - 8);
  const nowH = hours[hours.length - 1];
  const dataLo = Math.min(...pts.map((p) => p.y));
  const dataHi = Math.max(...pts.map((p) => p.y));
  const lastWhere = whereVsNormal(v, last.y);
  const summary = `${v.label}: ${pts.length} measurement${pts.length === 1 ? "" : "s"}, last ${fmtValue(last.y, v.digits)} ${v.unit} at hour ${last.h} (${WHERE[lastWhere]}), range ${fmtValue(dataLo, v.digits)} to ${fmtValue(dataHi, v.digits)}, normal ${fmtNormal(v)}`;

  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const r = svgRef.current?.getBoundingClientRect();
    if (!r) return;
    const sx = ((e.clientX - r.left) / r.width) * W;
    let best = 0;
    pts.forEach((p, i) => {
      if (Math.abs(x(p.h) - sx) < Math.abs(x(pts[best].h) - sx)) best = i;
    });
    setHover(best);
  };
  const hp = hover == null ? null : pts[hover];
  const hpWhere = hp ? whereVsNormal(v, hp.y) : "normal";
  const leftPct = hp ? (x(hp.h) / W) * 100 : 0;

  return (
    <div className="mini">
      <div className="mini-head">
        <span className="mini-name">{v.label}</span>
        <span className={`mini-val ${lastWhere !== "normal" ? "out" : ""}`}>
          {fmtValue(last.y, v.digits)} <span className="unit">{v.unit}</span>
        </span>
      </div>
      <div className="mini-plot">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={summary}
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
        >
          <rect className="nr" x={0} width={W} y={y(v.normal[1])} height={Math.max(y(v.normal[0]) - y(v.normal[1]), 1)} />
          {onset != null && onset <= xMax && <line x1={x(onset)} x2={x(onset)} y1={0} y2={H} className="on" />}
          <line x1={x(nowH)} x2={x(nowH)} y1={0} y2={H} className="nw" />
          {pts.length > 1 && <polyline className="ln" points={pts.map((p) => `${x(p.h)},${y(p.y)}`).join(" ")} />}
          {hp && <line x1={x(hp.h)} x2={x(hp.h)} y1={0} y2={H} className="hv" />}
          {pts.map((p, i) => (
            <circle key={p.h} className={`pt ${i === hover ? "hot" : ""}`} cx={x(p.h)} cy={y(p.y)} r={i === hover ? 4 : 2.4} />
          ))}
        </svg>
        {hp && (
          <div
            className="mini-tip"
            style={leftPct > 55 ? { right: `${100 - leftPct + 3}%` } : { left: `${leftPct + 3}%` }}
            aria-hidden="true"
          >
            <div className="mini-tip-h">ICU hour {hp.h}</div>
            <div className="mini-tip-v">
              {fmtValue(hp.y, v.digits)} {v.unit}
            </div>
            <div className={`mini-tip-n ${hpWhere !== "normal" ? "out" : ""}`}>{WHERE[hpWhere]}</div>
          </div>
        )}
      </div>
      <div className="mini-meta">
        <span>
          {pts.length}× · last at hour {last.h}
        </span>
        <span className="range">
          {fmtValue(dataLo, v.digits)}–{fmtValue(dataHi, v.digits)}
        </span>
      </div>
      <div className="mini-meta">{normal}</div>
    </div>
  );
}
