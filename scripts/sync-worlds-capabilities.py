"""Keep browser launch controls aligned with the actual CPU-only worker adapters."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers/worlds"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    runtimes = {}
    for name, classname in (
        ("astronex", "AstronexAdapter"),
        ("forgewm", "Adapter"),
        ("matrix_game", "Adapter"),
        ("sana_wm", "Adapter"),
        ("helixworld", "Adapter"),
        ("ltx25", "Adapter"),
    ):
        adapter = getattr(importlib.import_module(f"{name}.adapter"), classname)()
        runtimes[adapter.model_id] = adapter.capabilities
    # JSON is intentionally a separate file so the generated artifact has stable formatting.
    target = ROOT / "apps/web/src/worlds/core/runtimeCapabilities.json"
    content = json.dumps(runtimes, indent=2) + "\n"
    if args.check:
        if not target.exists() or json.loads(target.read_text()) != runtimes:
            raise SystemExit("Worlds browser capabilities are stale. Run scripts/sync-worlds-capabilities.py")
    else:
        target.write_text(content)


if __name__ == "__main__":
    main()
