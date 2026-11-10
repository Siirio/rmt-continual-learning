#!/usr/bin/env python3
"""Regenerate the deterministic example results package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rmt.reporting.example_data import generate_example_package  # noqa: E402
from rmt.reporting.report_generator import generate_results_markdown  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "examples/results")
    parser.add_argument("--seed", type=int, default=20261110)
    args = parser.parse_args()
    manifest = generate_example_package(args.output, args.seed)
    generate_results_markdown(args.output, args.output / "RESULTS.md")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

