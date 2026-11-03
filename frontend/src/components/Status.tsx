import { Level } from "../api";
import { StatusGlyph } from "./Icons";

export const LEVEL_WORD: Record<Level, string> = { ALERT: "Alert", WATCH: "Watch", OK: "OK" };

/** Status as shape + word (colour is a third, redundant cue). `off` = discharged. */
export function Status({ level, off = false, size = 14 }: { level: Level; off?: boolean; size?: number }) {
  if (off) return <span className="status st-off">Discharged</span>;
  return (
    <span className={`status st-${level.toLowerCase()}`}>
      <StatusGlyph level={level} size={size} />
      {LEVEL_WORD[level]}
    </span>
  );
}

/** The same status mounted on a label board. */
export function StatusChip({ level, off = false }: { level: Level; off?: boolean }) {
  if (off) return <span className="chip chip-off">Discharged</span>;
  return (
    <span className={`chip chip-${level.toLowerCase()}`}>
      <StatusGlyph level={level} size={14} />
      {LEVEL_WORD[level]}
    </span>
  );
}

/** Level of one hour, from its risk and alert flag at threshold tau (Watch = risk ≥ ½τ). */
export const levelAt = (risk: number, alert: boolean, tau: number): Level => (alert ? "ALERT" : risk >= tau / 2 ? "WATCH" : "OK");

/** "Bed 16" → "16": the exhibit's catalogue number, identical on every screen. */
export const bedNumber = (displayName: string) => displayName.replace(/^Bed\s*/i, "");
