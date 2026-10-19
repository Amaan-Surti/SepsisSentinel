import { Level } from "../api";

const LABEL: Record<Level, string> = { ALERT: "Alert", WATCH: "Watch", OK: "OK" };
const GLYPH: Record<Level, string> = { ALERT: "▲", WATCH: "●", OK: "–" };

export function LevelChip({ level }: { level: Level }) {
  return (
    <span className={`chip chip-${level.toLowerCase()}`}>
      <span aria-hidden="true">{GLYPH[level]}</span> {LABEL[level]}
    </span>
  );
}
