"""Capture dashboard screenshots with headless Chrome over the DevTools protocol.

Usage (app running at http://localhost:5173 or :8080):
    .venv\\Scripts\\python.exe reports/screenshots/capture.py --base http://localhost:8080

Each shot opens a deep link (?t=<hour>&truth=1), waits for the API data and saves a JPEG.
Uses only the local Chrome install and the `websockets` package already in the venv.
"""

import argparse
import base64
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from websockets.sync.client import connect

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
OUT = Path(__file__).resolve().parent


class Page:
    """Minimal CDP client for one tab."""

    def __init__(self, ws_url: str) -> None:
        self.ws = connect(ws_url, max_size=50 * 2**20)
        self.n = 0

    def call(self, method: str, **params: Any) -> Dict[str, Any]:
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def js(self, expr: str) -> Any:
        res = self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        return res.get("result", {}).get("value")

    def wait_for(self, expr: str, timeout: float = 15.0) -> None:
        end = time.time() + timeout
        while time.time() < end:
            if self.js(expr):
                return
            time.sleep(0.2)
        raise TimeoutError(expr)


def capture(
    page: Page, url: str, out: Path, width: int, height: int,
    ready: str, steps: Optional[List[str]] = None, full: bool = False,
) -> None:
    page.call("Emulation.setDeviceMetricsOverride", width=width, height=height,
              deviceScaleFactor=1, mobile=False)
    page.call("Page.navigate", url=url)
    page.wait_for("document.readyState === 'complete'")
    page.wait_for(ready)
    for s in steps or []:
        page.js(s)
        time.sleep(0.4)
    time.sleep(1.2)  # chart animations and debounced fetches
    params: Dict[str, Any] = {"format": "jpeg", "quality": 88}
    if full:
        h = page.js("document.documentElement.scrollHeight")
        params["clip"] = {"x": 0, "y": 0, "width": width, "height": h, "scale": 1}
        params["captureBeyondViewport"] = True
    data = page.call("Page.captureScreenshot", **params)["data"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(base64.b64decode(data))
    print("saved", out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture dashboard screenshots")
    parser.add_argument("--base", default="http://localhost:8080")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--hour", type=int, default=30)
    parser.add_argument("--bed", default="Bed 16")
    parser.add_argument("--legacy", action="store_true",
                        help="Pre-P9 UI without deep links: drive the replay controls instead")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    out = Path(args.out)

    port = 9333
    profile = tempfile.mkdtemp(prefix="sepsis-shots-")
    chrome = subprocess.Popen([
        CHROME, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={profile}",
        "--hide-scrollbars", "--no-first-run", "--disable-extensions", "about:blank",
    ])
    try:
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list"))
                ws = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
                break
            except (OSError, StopIteration):
                time.sleep(0.2)
        page = Page(ws)
        page.call("Page.enable")
        rows_ready = "document.querySelectorAll('tbody tr').length >= 24"
        bed_click = (
            "[...document.querySelectorAll('tbody tr')]"
            f".find(r => r.textContent.includes({json.dumps(args.bed)})).click()"
        )
        w, h, sfx = args.width, args.height, args.suffix
        if args.legacy:
            step = ("(() => { const b = [...document.querySelectorAll('button')]"
                    ".find(b => b.textContent.trim() === '+1h'); "
                    f"for (let i = 0; i < {args.hour}; i++) b.click(); }})()")
            truth = "document.querySelector('input[type=checkbox]').click()"
            capture(page, f"{args.base}/", out / f"ward{sfx}.jpg", w, h, rows_ready, [step])
            capture(page, f"{args.base}/", out / f"patient{sfx}.jpg", w, h, rows_ready,
                    [step, truth, bed_click])
            capture(page, f"{args.base}/model", out / f"model{sfx}.jpg", w, h,
                    "document.querySelectorAll('table').length > 0")
            return
        q = f"?t={args.hour}&truth=1"
        capture(page, f"{args.base}/?t={args.hour}", out / f"ward{sfx}.jpg", w, h, rows_ready)
        capture(page, f"{args.base}/{q}", out / f"patient{sfx}.jpg", w, h, rows_ready,
                [bed_click])
        capture(page, f"{args.base}/model", out / f"model{sfx}.jpg", w, h,
                "document.querySelectorAll('table').length > 1")
    finally:
        chrome.terminate()


if __name__ == "__main__":
    main()
