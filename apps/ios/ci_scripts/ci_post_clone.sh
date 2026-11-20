#!/bin/bash
set -euo pipefail

if ! command -v xcodegen >/dev/null 2>&1; then
  brew install xcodegen
fi
# Xcode Cloud images have no Node; the bundled Cesium globe is built from apps/web.
if ! command -v node >/dev/null 2>&1; then
  brew install node@22
  export PATH="$(brew --prefix node@22)/bin:$PATH"
fi
if ! command -v pnpm >/dev/null 2>&1; then
  npm install -g pnpm@10
fi
cd "$(dirname "$0")/.."
(cd ../.. && pnpm install --frozen-lockfile)
bash ci_scripts/build_globe.sh
xcodegen generate
