#!/usr/bin/env python3
"""Aggregate completed run JSON files into a CSV metric table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/raw"))
    parser.add_argument("--output", type=Path, default=Path("results/tables/run_metrics.csv"))
    args = parser.parse_args()
    rows = []
    for path in sorted(args.input.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "completed":
            continue
        specification = payload["specification"]
        result = payload["result"]
        stages = result.get("stages", [])
        rows.append(
            {
                "run_id": specification["run_id"],
                "method": specification["method"],
                "seed": specification["seed"],
                "stages": len(stages),
                "status": payload["status"],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["run_id", "method", "seed", "stages", "status"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

