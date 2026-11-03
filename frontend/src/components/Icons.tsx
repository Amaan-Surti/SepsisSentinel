import type { ReactNode } from "react";
import { Level } from "../api";

/** Small authored icons: one 16-unit grid, 1.75 stroke, currentColor. */
type IconProps = { size?: number; className?: string };

function Svg({ size = 16, className, children }: IconProps & { children: ReactNode }) {
  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.75}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      {children}
    </svg>
  );
}

export const PlayIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M5 3.5v9l7-4.5z" fill="currentColor" />
  </Svg>
);

export const PauseIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M5.5 3.5v9M10.5 3.5v9" strokeWidth={2.25} />
  </Svg>
);

export const ResetIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M3.5 3.5v9M12.5 3.5 6.5 8l6 4.5z" />
  </Svg>
);

export const BackIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M10 3.5 5.5 8l4.5 4.5" />
  </Svg>
);

export const ForwardIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M6 3.5 10.5 8 6 12.5" />
  </Svg>
);

/** Status shapes differ by form as well as colour: triangle, circle, bar. */
export function StatusGlyph({ level, size = 12 }: { level: Level; size?: number }) {
  return (
    <Svg size={size} className={`status-glyph glyph-${level.toLowerCase()}`}>
      {level === "ALERT" && <path d="M8 2.5 14 13H2z" fill="currentColor" strokeWidth={1.25} />}
      {level === "WATCH" && <circle cx="8" cy="8" r="4.75" fill="currentColor" strokeWidth={0} />}
      {level === "OK" && <path d="M3.5 8h9" strokeWidth={2.25} />}
    </Svg>
  );
}

/** Upward marker for a factor that raises risk. */
export const RaiseIcon = (p: IconProps) => (
  <Svg {...p}>
    <path d="M8 3v10M4 7l4-4 4 4" />
  </Svg>
);
