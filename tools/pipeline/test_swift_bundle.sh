#!/usr/bin/env bash
# Offline Linux Swift -> Python bundle validation. Run from any directory.
set -euo pipefail
repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if ! command -v swift >/dev/null && [[ -f "$HOME/hw-tools/swift/env.sh" ]]; then
  . "$HOME/hw-tools/swift/env.sh"
fi
command -v swift >/dev/null || { echo "Swift must be on PATH" >&2; exit 1; }
export TMPDIR="${TMPDIR:-$repository/.scratch/tmp}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$repository/.scratch/uv-cache}"
mkdir -p "$TMPDIR" "$UV_CACHE_DIR"
cd "$repository/tools/pipeline"
uv run --offline --no-sync pytest -q tests/test_swift_bundle.py "$@"
