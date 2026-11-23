#!/usr/bin/env bash
set -euo pipefail

# Run from any directory; keep compiler caches and test files inside the worktree.
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
case "${1:-all}" in
  all|format) filter=() ;;
  replay) filter=(--filter 'replay|fakeSource') ;;
  seeding) filter=(--filter 'seedUses|frameRejects|replayYardSmall') ;;
  coverage) filter=(--filter 'coverage|Coverage') ;;
  fake) filter=(--filter 'fake|rendererProtocol') ;;
  math) filter=(--filter 'Projection|camera|Covariance|covariance|surfel|fusionDispatch|gaussianRejects') ;;
  lifecycle) filter=(--filter 'rendererCaps|qualityRecovers|fiveMinute|trackingLoss|rendererProtocol') ;;
  export) filter=(--filter 'PLY|export|rendererProtocol') ;;
  *) echo "Usage: bash $0 [all|replay|seeding|coverage|fake|math|lifecycle|export|format]" >&2; exit 2 ;;
esac

if ! command -v swift >/dev/null && [[ -f "$HOME/hw-tools/swift/env.sh" ]]; then
  source "$HOME/hw-tools/swift/env.sh"
fi
command -v swift >/dev/null
command -v heavy >/dev/null
mkdir -p .tmp-test .swift-scratch
trap 'rm -rf -- ./.tmp-test ./.swift-scratch' EXIT
export TMPDIR="$PWD/.tmp-test"
export CLANG_MODULE_CACHE_PATH="$PWD/.swift-scratch/clang-cache"
export SWIFTPM_MODULECACHE_OVERRIDE="$PWD/.swift-scratch/module-cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "${1:-all}" == format ]]; then
  pnpm format:check
  exit
fi
HEAVY_CPUS=2 heavy swift test --package-path apps/ios/Modules/LiveSplat \
  --scratch-path ./.swift-scratch/LiveSplat --cache-path ./.swift-scratch/cache \
  --config-path ./.swift-scratch/config --security-path ./.swift-scratch/security \
  --disable-sandbox --jobs 1 --no-parallel "${filter[@]}"
