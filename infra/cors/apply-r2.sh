#!/usr/bin/env bash
# Apply a CORS document to an R2 bucket, with the real origin substituted.
#
#   AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=... \
#     infra/cors/apply-r2.sh <account-id> <bucket> <origin> [<origin>...] [-- <document>]
#
# e.g. infra/cors/apply-r2.sh a1b2c3 twin-assets https://twin.example.com -- upload.json
#
# ONE bucket, ONE document. R2 (like S3) keeps a single CORS configuration per bucket and
# `put-bucket-cors` replaces it wholesale, so a bucket in two roles needs its two rules
# combined into one document -- which is what production.json is, and what a one-bucket
# deployment still applies.
#
# A split deployment does not need the compromise. The private bucket takes upload.json
# (browser PUTs to presigned URLs under `captures/`) and the public bucket takes
# tiles.json (CesiumJS reading `tileset.json` and `.glb`), because after the split those
# really are two buckets with one role each. The default is still production.json, so a
# caller that has one bucket gets the combined document without asking for it.
#
# UNVERIFIED, and this is the script that will find out. Two claims below have never met
# the real R2 API:
#
#   1. That R2 rejects a narrow `AllowedHeaders: ["content-type"]` and requires `["*"]`.
#      That has been asserted, not evidenced, and it could not be checked from the
#      environment these rules were written in (developers.cloudflare.com unreachable).
#      The documents use `["*"]`, which is accepted either way, so the deployment does
#      not depend on the answer -- but the claim should not be repeated as fact.
#   2. That R2's S3 API accepts `put-bucket-cors` at all, on this endpoint, with these
#      credentials. `apps/api/tests/test_cors_rules.py` proves the documents are shape
#      valid to botocore and agree with each other. Nothing proves a provider accepts
#      them. This script prints what the bucket reports back afterwards for that reason:
#      read it, and correct docs/DEPLOYMENT.md with what R2 actually said.
set -euo pipefail

if [ "$#" -lt 3 ]; then
  sed -n '2,8p' "$0" >&2
  exit 2
fi

account_id="$1"
bucket="$2"
shift 2

# Origins up to `--`; the document after it. Origins are variadic, so the separator is
# what keeps a document from being mistaken for one more origin.
origins=()
document="production.json"
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--" ]; then
    shift
    [ "$#" -eq 1 ] || { echo "expected exactly one document after --" >&2; exit 2; }
    document="$1"
    shift
  else
    origins+=("$1")
    shift
  fi
done
[ "${#origins[@]}" -gt 0 ] || { echo "no origins given" >&2; exit 2; }
set -- "${origins[@]}"

here="$(cd "$(dirname "$0")" && pwd)"
endpoint="https://${account_id}.r2.cloudflarestorage.com"
rendered="$(mktemp)"
trap 'rm -f "$rendered"' EXIT

# The committed document carries a placeholder origin. Swapping it here keeps the origin
# out of git and keeps the rules in it.
python3 - "$here/$document" "$rendered" "$@" <<'PY'
import json
import sys

source, destination, *origins = sys.argv[1:]
document = json.load(open(source))
for rule in document["CORSRules"]:
    rule["AllowedOrigins"] = origins
json.dump(document, open(destination, "w"), indent=2)
print(f"applying {len(document['CORSRules'])} rules from {source} for {origins}")
PY

# R2 wants region "auto"; a real region name makes botocore sign for a host that is not
# there. The same value belongs in OBJECT_STORAGE_REGION.
aws s3api put-bucket-cors \
  --endpoint-url "$endpoint" \
  --region auto \
  --bucket "$bucket" \
  --cors-configuration "file://$rendered"

echo "--- what the bucket reports back ---"
aws s3api get-bucket-cors \
  --endpoint-url "$endpoint" \
  --region auto \
  --bucket "$bucket"
