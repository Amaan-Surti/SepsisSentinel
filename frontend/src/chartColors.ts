/** Chart colours read from the CSS tokens in styles.css, so charts and CSS never drift. */
const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export const C = {
  get ink() {
    return css("--ink");
  },
  get muted() {
    return css("--ink-muted");
  },
  get rule() {
    return css("--rule");
  },
  get wall() {
    return css("--wall");
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
  get gru() {
    return css("--gru");
  },
  get annexInk() {
    return css("--annex-ink");
  },
};

/** Axis tick sizes for a laptop viewed at arm's length (the type scale's two smallest steps). */
export const TICK = { main: 14, mini: 14 };
