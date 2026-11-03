/** Chart colours read from the CSS tokens in styles.css, so charts and CSS never drift. */
const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export const C = {
  get text() {
    return css("--text");
  },
  get muted() {
    return css("--muted");
  },
  get grid() {
    return css("--line");
  },
  get teal() {
    return css("--teal");
  },
  get alert() {
    return css("--alert");
  },
  get alertInk() {
    return css("--alert-ink");
  },
  get watch() {
    return css("--watch");
  },
  get dev() {
    return css("--dev");
  },
};

/** Axis tick sizes chosen for a classroom projector. */
export const TICK = { main: 13, mini: 12 };
