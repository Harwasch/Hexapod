# Bucket CORS rules

Two roles, two rules. A bucket has exactly **one** CORS configuration, so a bucket
in both roles gets one document containing both rules -- that is what
`dev-minio.xml` and `production.json` are.

| File              | Role                                                    | Why                                                                                                                                                          |
| ----------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `upload.json`     | browser PUTs to presigned URLs                          | `ExposeHeaders: ["ETag"]` is load-bearing: without it the browser reads `etag === null` off the part response and a multipart upload can never be completed  |
| `tiles.json`      | CesiumJS reading `tileset.json` and `.glb` cross-origin | read-only, long `MaxAgeSeconds`, range-request headers exposed                                                                                               |
| `dev-minio.xml`   | both, for the single dev bucket                         | `mc cors set` takes XML; everything else takes JSON                                                                                                          |
| `production.json` | both, for a single bucket in both roles                 | a deployment with one bucket: uploads land under `captures/` and published tiles under `sites/` in the same place, and R2 keeps one CORS document per bucket |

Applying `upload.json` and then `tiles.json` to the **same** bucket is the mistake
this directory exists to make hard: the second call replaces the first, and browser
multipart uploads stop being completable the moment you do it. That is why a
one-bucket deployment gets one combined document, and why
`apps/api/tests/test_cors_rules.py` asserts it has not drifted from the two it
combines.

**A split deployment applies one document per bucket instead**, and the combined one
to neither. Setting `OBJECT_STORAGE_PUBLIC_BUCKET` gives each bucket a single role —
uploads and run outputs stay in the private bucket, published tiles go to the public
one — so `upload.json` goes on the first and `tiles.json` on the second, each the
only document its bucket ever has. `provision.yml` does exactly that, and
`apply-r2.sh` takes the document to apply after a `--`:

```bash
infra/cors/apply-r2.sh <account> twin-assets https://twin.example.com -- upload.json
infra/cors/apply-r2.sh <account> twin-public https://twin.example.com -- tiles.json
```

The combination was always a workaround for having one bucket, not a preference.

`AllowedOrigins` are placeholders -- replace `https://twin.example.com` with your
real origin. `infra/cors/apply-r2.sh` does that substitution for you at apply time,
so the origin never has to be committed. The production document deliberately
carries no `localhost` origin; the dev documents do.

The tests also check that the XML and the JSON documents still describe the same
rules, and that the ETag exposure has not been dropped.

See `docs/DEPLOYMENT.md` for the commands that apply these to MinIO and to R2.
