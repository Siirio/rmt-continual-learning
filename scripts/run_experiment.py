#!/usr/bin/env python3
"""Execute one planned experiment or the complete 19-run schedule."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rmt.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(["run", *sys.argv[1:]]))

