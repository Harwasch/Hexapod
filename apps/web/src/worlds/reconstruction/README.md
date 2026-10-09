# Worlds reconstruction

The source picker, browser video-frame extraction, selection, TAR export and local
PLY/SPZ viewer work without a GPU or remote service. No reconstruction is invented:
the TAR holds the selected image/video bytes and `manifest.json`, including native
camera poses only when they were actually supplied. Generated video can violate
multi-view geometry. Prefer overlapping, static views with camera translation.

The independent Spark viewer reuses the renderer already installed for Hexapod's
standalone scan viewer. PLY point clouds use Three's PLY loader. Nothing creates an
Earth capture or publishes a map site. The legacy `photo-reconstruct` recipe is
deliberately not submitted: it publishes Earth sites and specifies Modal stages.

## Optional reconstruction worker contract

`app.worlds.reconstruction.router` exposes authenticated endpoints under
`/api/v1/worlds/reconstructions`. A configured gateway normally runs on RunPod;
any independently hosted implementation of the contract can be used. The packaged
[Mirror engine](../../../../../workers/worlds/reconstruction/README.md) provides
real pinned inference, Gaussian PLY, colored points and an observed-depth GLB export,
with bounded authenticated jobs and confirmed deletion. It provisions no compute.
No GPU run or reconstruction-quality measurement was performed during implementation.

Operator settings, server-side only:

```
WORLD_RECONSTRUCTION_GATEWAY_URL=https://your-worker.example
WORLD_RECONSTRUCTION_GATEWAY_TOKEN=your-worker-token
```

HTTPS is required for remote workers. HTTP is accepted only on loopback for a
local process. The gateway must implement:

- `POST /reconstructions`: multipart `files` entries and a JSON `metadata` form
  field. Accept 1–120 JPEG/PNG/WebP images or MP4/WebM/MOV recordings, at most
  64 MiB total. Submission should enqueue quickly and return a job.
- `GET /reconstructions/{id}`: current job state, including newly signed download
  URLs when necessary.
- `DELETE /reconstructions/{id}`: cancel any remaining work and remove the job's
  input, partial and output media. Return 204, or 200 with `{"deleted":true}`, only
  after removal completes. An unsupported or unconfirmed deletion is shown as a
  failure; the app keeps the job reference and does not claim remote cleanup.

Example response:

```json
{
  "id": "random-job-id",
  "status": "running",
  "progress": 0.35,
  "stage": "Estimating camera poses",
  "artifacts": []
}
```

Status is `queued`, `running`, `completed` or `failed`. Progress is an optional
measured fraction in `[0,1]`; omit it if unknown. Finished `artifacts` entries
contain `format` (`ply`, `spz`, `glb`, `gltf`, `point-cloud`) and a short-lived HTTPS
`url`. Only advertise formats actually produced. GLTF dependencies should be
packaged or exposed as a coherent export; prefer self-contained GLB. CORS must
allow the Worlds origin to load PLY/SPZ artifacts into the viewer. Downloads remain
available if CORS prevents inline loading.

Completed jobs may also return optional `diagnosticsUrl` and `camerasUrl` signed
downloads. The UI exposes these after refreshing status. Mirror's diagnostics are
prediction self-consistency and source heuristics, not ground-truth accuracy;
camera scale is model-relative. Links expire and refresh with the job status.

Recognized stage labels are `Queued`, `Preparing sources`, `Extracting frames`,
`Estimating camera poses`, `Estimating depth`, `Reconstructing scene`, `Training
splat`, `Exporting geometry`, `Completed` and `Failed`. Other stage strings are
omitted to avoid exposing paths, prompts or raw worker trace information.

The API strips prompts, project IDs, user filenames and arbitrary metadata before
forwarding. It sends neutral numbered filenames and valid optional per-frame
`timestampMs` / 12- or 16-value `cameraPose`. Worker credentials are added only by
the API, never included in media metadata or returned to the browser. The worker
must protect endpoints, enforce retention, remove temporary input data and avoid
logging user media. An absent/invalid gateway is reported as unavailable, while
local export and viewing continue to work.

Images are decoded with a 16-megapixel limit, oriented and rewritten as PNG without
EXIF, GPS, XMP or other embedded metadata before remote submission. Videos are
forwarded as selected, including any embedded metadata; extract frames locally when
only image content is needed. Multipart input is bounded during streaming before
spooling can exceed the 64 MiB media limit plus 1 MiB form overhead. Normalized
outbound media is also limited to 64 MiB. Temporary API upload files close on parse
errors, while remote partial-upload cleanup and retention remain gateway duties.

The embedded viewer caps files/download streams at 128 MiB and geometry at two
million points. PLY list/mesh properties are refused before parsing. Gzip SPZ v2/v3
is checked for its declared point count and a 256 MiB decompressed limit. Larger
or different-format geometry can still be downloaded for a desktop viewer.

## Official reconstruction candidates checked 2026-10-08

Mirror is now packaged in this repository; the others remain comparison candidates.
None has a measured GPU quality result in this workspace:

- [HunyuanWorld-Mirror](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror):
  the official README describes feed-forward camera/depth/point-cloud and Gaussian
  prediction; its example exporter writes COLMAP data and `gaussians.ply`.
  Review its project license and checkpoint terms before selecting a deployment.
- [VGGT](https://github.com/facebookresearch/vggt): camera/depth/point prediction
  and COLMAP export can feed gsplat training. The README distinguishes the original
  noncommercial weights from the gated
  [VGGT-1B-Commercial checkpoint](https://huggingface.co/facebook/VGGT-1B-Commercial)
  and its commercial-use license restrictions.
- [InstantSplat](https://github.com/NVlabs/InstantSplat): official sparse-view
  Gaussian reconstruction implementation. Its repository and constituent model/
  rasterizer licenses must be checked for the intended deployment.

No single method is assumed to reconstruct dynamic, inconsistent neural imagery.
For comparison, record registered-view fraction, reprojection error, held-out
render quality and revisited-region alignment on the same selected frames.
