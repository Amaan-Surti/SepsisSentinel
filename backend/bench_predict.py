"""Time POST /api/predict for a 70-hour stay against a running API.

Usage: .venv\\Scripts\\python.exe backend/bench_predict.py --url http://127.0.0.1:8000 --n 30
The stay is the first 70 hours of the longest demo bed (raw columns only).
"""

import argparse
import json
import math
import statistics
import time
import urllib.request
from pathlib import Path

import pandas as pd

from sepsis.constants import RAW_COLUMNS

DEMO = Path(__file__).resolve().parents[1] / "data" / "demo"


def payload(hours: int = 70) -> bytes:
    rows = pd.read_parquet(DEMO / "demo_rows.parquet")
    pid = rows.groupby("patient_id").size().idxmax()
    stay = rows[rows["patient_id"] == pid].sort_values("hour").head(hours)
    out = []
    for _, r in stay.iterrows():
        d = {"hour": int(r["hour"])}
        for c in RAW_COLUMNS:
            if c == "SepsisLabel":
                continue
            v = float(r[c])
            d[c] = None if math.isnan(v) else v
        out.append(d)
    if len(out) != hours:
        raise ValueError(f"Longest demo stay has only {len(out)} hours")
    return json.dumps({"rows": out}).encode()


def paired(rows: list, n: int) -> None:
    """Alternate the two services' predict_rows so CPU boost and noise hit both alike."""
    from backend.app.service import SepsisService
    from backend.app.settings import Settings

    services = {k: SepsisService(Settings(model_kind=k)) for k in ("lgbm", "ensemble")}
    times: dict = {k: [] for k in services}
    for svc in services.values():
        for _ in range(3):
            svc.predict_rows(rows)
    for _ in range(n):
        for k, svc in services.items():
            t0 = time.perf_counter()
            svc.predict_rows(rows)
            times[k].append((time.perf_counter() - t0) * 1000)
    out = {k: round(statistics.median(v), 1) for k, v in times.items()}
    out["gru_overhead_ms"] = round(out["ensemble"] - out["lgbm"], 1)
    print(json.dumps({"mode": "in-process, interleaved", "hours": len(rows), "n": n,
                      "median_ms": out}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark /api/predict")
    # 127.0.0.1, not localhost: on Windows localhost may try ::1 first and add ~2 s
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--hours", type=int, default=70)
    parser.add_argument(
        "--in-process", action="store_true",
        help="Interleave LightGBM and ensemble calls in one process (paired, no HTTP)",
    )
    args = parser.parse_args()
    body = payload(args.hours)
    if args.in_process:
        paired(json.loads(body)["rows"], args.n)
        return
    with urllib.request.urlopen(f"{args.url}/api/health") as r:
        health = json.load(r)

    def call() -> float:
        req = urllib.request.Request(
            f"{args.url}/api/predict", data=body, headers={"Content-Type": "application/json"}
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req) as r:
            r.read()
        return (time.perf_counter() - t0) * 1000

    for _ in range(3):  # warm-up (SHAP explainer, torch threads)
        call()
    times = sorted(call() for _ in range(args.n))
    result = {
        "model_kind": health["model_kind"],
        "model_version": health["model_version"],
        "hours": args.hours,
        "n": args.n,
        "median_ms": round(statistics.median(times), 1),
        "p90_ms": round(times[int(0.9 * (len(times) - 1))], 1),
        "min_ms": round(times[0], 1),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
