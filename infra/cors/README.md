# Bucket CORS rules

Two roles, two rules. A bucket has exactly **one** CORS configuration, so a bucket
in both roles gets one document containing both rules -- that is what
`dev-minio.xml` is.

| File            | Role                                                    | Why                                                                                                                                                         |
| --------------- | ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `upload.json`   | browser PUTs to presigned URLs                          | `ExposeHeaders: ["ETag"]` is load-bearing: without it the browser reads `etag === null` off the part response and a multipart upload can never be completed |
| `tiles.json`    | CesiumJS reading `tileset.json` and `.glb` cross-origin | read-only, long `MaxAgeSeconds`, range-request headers exposed                                                                                              |
| `dev-minio.xml` | both, for the single dev bucket                         | `mc cors set` takes XML; everything else takes JSON                                                                                                         |

`AllowedOrigins` here are placeholders -- replace `https://twin.example.com` with
your real origin. `apps/api/tests/test_cors_rules.py` checks that the XML and the
two JSON documents still describe the same rules, and that the ETag exposure has
not been dropped.

See `docs/DEPLOYMENT.md` for the commands that apply these to MinIO and to R2.
