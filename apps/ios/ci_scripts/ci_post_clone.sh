#!/bin/bash
set -euo pipefail

if ! command -v xcodegen >/dev/null 2>&1; then
  brew install xcodegen
fi
cd "$(dirname "$0")/.."
xcodegen generate
