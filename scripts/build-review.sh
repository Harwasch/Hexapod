#!/usr/bin/env bash
# Build the review page: review-artifact from the repo, then the 3-D viewer
# of the concept skeletons injected into the Vision tab.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/opt/hw-py/bin/python
review-artifact --check
review-artifact
$PY scripts/export_models.py
$PY scripts/inject_viewer.py
