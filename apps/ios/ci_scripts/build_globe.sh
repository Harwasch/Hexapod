#!/bin/bash
# Local dependencies only. CI provisions Node/pnpm before invoking this script.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$root"
node apps/web/scripts/generate-globe-bridge.mjs --check
GLOBE_EMBED=1 pnpm --filter @twin/web build
node apps/web/scripts/copy-globe-embed.mjs
