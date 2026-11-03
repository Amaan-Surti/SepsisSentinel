import { Level } from "../api";
import { StatusGlyph } from "./Icons";

const LABEL: Record<Level, string> = { ALERT: "Alert", WATCH: "Watch", OK: "OK" };

/** Status is always shape + word, never colour alone. */
export function LevelChip({ level }: { level: Level }) {
  return (
    <span className={`chip chip-${level.toLowerCase()}`}>
      <StatusGlyph level={level} />
      {LABEL[level]}
    </span>
  );
}
