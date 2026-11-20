#!/usr/bin/env bash
# Offline integration gate using installed toolchains and local dependencies only.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .tmp-test .swift-scratch
export TMPDIR="$PWD/.tmp-test" UV_CACHE_DIR="$PWD/.tmp-test/uv" UV_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HEAVY_CPUS=2
export npm_config_workspace_concurrency=1
export NODE_OPTIONS="${NODE_OPTIONS:---max-old-space-size=8192}"
trap 'rm -rf .tmp-test .swift-scratch' EXIT
if ! command -v swift >/dev/null; then
  . "$HOME/hw-tools/swift/env.sh"
fi
for package in apps/ios/Modules/* apps/ios/App; do
  if [ -f "$package/Package.swift" ]; then
    heavy swift test --package-path "$package" --scratch-path "$PWD/.swift-scratch/$(basename "$package")" --jobs 1 --disable-sandbox
  fi
done
rm -rf .swift-scratch .tmp-test
mkdir -p .tmp-test
python3 -m unittest discover -s apps/ios/ci_scripts
python3 -m unittest discover -s .github/scripts
heavy pnpm format:check
heavy pnpm lint
heavy pnpm --workspace-concurrency=1 -r typecheck
heavy pnpm --workspace-concurrency=1 -r --filter './packages/**' --filter './apps/web' test --pool=threads --maxWorkers=1
heavy pnpm contracts:check
for project in apps/api tools/analysis tools/pipeline; do
  (
    cd "$project"
    uv run --no-sync ruff check .
    uv run --no-sync ruff format --check .
    heavy uv run --no-sync mypy .
    if [ "$project" = apps/api ] && [ -z "${TEST_DATABASE_URL:-}" ]; then
      heavy uv run --no-sync pytest -q --without-db
    else
      heavy uv run --no-sync pytest -q
    fi
  )
done
