"""Write the OpenAPI document to a file. Used to generate the typed frontend client."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app.main import create_app


def main(argv: list[str]) -> int:
    target = Path(argv[1]) if len(argv) > 1 else Path("openapi.json")
    document = create_app().openapi()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(f"wrote {target}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
