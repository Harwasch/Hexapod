#!/usr/bin/env bash
# Apply the production CORS document to the R2 bucket, with the real origin substituted.
#
#   AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=... \
#     infra/cors/apply-r2.sh <account-id> <bucket> <origin> [<origin>...]
#
# e.g. infra/cors/apply-r2.sh a1b2c3 twin-assets https://twin.example.com
#
# ONE bucket, ONE document. R2 (like S3) keeps a single CORS configuration per bucket and
# `put-bucket-cors` replaces it wholesale. This deployment's bucket is in both roles --
# the browser PUTs capture sources to presigned URLs under `captures/`, and CesiumJS
# reads published tiles under `sites/` -- because the API has one OBJECT_STORAGE_BUCKET
# setting and both paths use it. So both rules go in one document: production.json.
# Applying upload.json and then tiles.json to the same bucket would leave only the
# second, and multipart upload would stop being completable the moment you did it.
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

here="$(cd "$(dirname "$0")" && pwd)"
endpoint="https://${account_id}.r2.cloudflarestorage.com"
rendered="$(mktemp)"
trap 'rm -f "$rendered"' EXIT

# The committed document carries a placeholder origin. Swapping it here keeps the origin
# out of git and keeps the rules in it.
python3 - "$here/production.json" "$rendered" "$@" <<'PY'
import json
import sys

source, destination, *origins = sys.argv[1:]
document = json.load(open(source))
for rule in document["CORSRules"]:
    rule["AllowedOrigins"] = origins
json.dump(document, open(destination, "w"), indent=2)
print(f"applying {len(document['CORSRules'])} rules for {origins}")
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
