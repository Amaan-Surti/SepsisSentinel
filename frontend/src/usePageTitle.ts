import { useEffect } from "react";

/** Per-route document title, so tabs and screen readers name the page. */
export function usePageTitle(title: string | null) {
  useEffect(() => {
    document.title = title ? `${title} · SepsisSentinel` : "SepsisSentinel";
  }, [title]);
}
