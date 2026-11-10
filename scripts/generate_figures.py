#!/usr/bin/env python3
"""Generate all PNG and SVG scientific figures from a result package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rmt.reporting.figures import generate_all_figures  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "examples/results")
    parser.add_argument("--output", type=Path, default=ROOT / "examples/figures")
    args = parser.parse_args()
    paths = generate_all_figures(args.input, args.output)
    print(f"Generated {len(paths)} files in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

