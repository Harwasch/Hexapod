# ADR 0013 — Site frames and epochs: a second capture of a site is a new epoch of it

**Status:** accepted · **Date:** 2026-10-19 · **Decision record:** D3 of the splat-pipeline build

## Context

Until now one capture made one site (`app/worker/registration.py`), and a re-run of that capture
rewrote the site's splat asset in place to point at the new tiles. Two things followed. A second
capture of the same field became a second, unrelated site with its own origin, so the two could not
be compared. And a re-run left the first run's tiles referenced by nothing: what a site showed
yesterday could not be shown again. The Living World is meant to show change over time, which needs
the opposite of both.

## Decision

1. **A site has one fixed frame** (`site_frames`, 1:1 with `sites`, never updated): the WGS84
   origin its first registered capture was placed at, with its height datum (`msl-unverified` for
   every frame today: GPS altitude is above the geoid and no geoid model is applied) and its UTM
   zone. It is a table, not `sites.metadata`, because `PATCH /sites` replaces metadata wholesale.
2. **Each capture of a site is an epoch** (`epochs`): ordinal, the capture, the run behind its
   current revision, `observed_at` (with where it came from, first that applies: the run
   manifest's tz-aware `capturedAtUtc`, labelled `manifest`; the capture row's `captured_at`; the
   manifest's naive `capturedAt`; the upload time, labelled `created`. `capturedAtUtc` is read
   before the capture row and is never copied into `captures.captured_at`, where it would pass
   for a date a person gave), its footprint, platform (detected from the frames since M3a, ADR
   0014; `phone-video` or `unknown` before), status (`published`; `rejected` and `superseded` are reserved for M3's gates), how it found
   its site (`explicit`, `spatial`, `new-site`), alignment (M3) and quality. One capture is one
   epoch; `captures.epoch_id`, for several captures of one day, is future work. The run, footprint
   and quality describe the epoch's current revision, so a re-run replaces them; what they said
   before is appended to the epoch's `history` (JSONB, append-only, oldest first: `jobId`,
   `footprint`, `quality`, `replacedAt`, `replacedByJobId`), so no earlier run's view is lost.
3. **An asset is one tileset of one epoch and one source kind** (`assets.epoch_id`,
   `assets.source_kind`: `observed`, or `inferred` from M5). A re-run of the same capture inserts a
   new asset revision of its epoch and supersedes the old one by setting its `valid_to` (and
   `default_visible = false`) — the only two writes ever made to it. Its URL is never rewritten,
   with one narrow exception (below: a sidecar attach). The epoch's current asset is its newest
   observed asset with no `valid_to`. A republish onto the URL the epoch already shows is no
   revision: generations are content-hashed.
4. **The site is resolved when the job starts**, because the pipeline has to place the capture in
   the frame while it runs (`services/epochs.frame_for_run`): a re-run keeps its epoch's frame; a
   capture that names a site (`siteId` on capture create and on the phone route, kept in
   `captures.metadata.siteId`) uses that site's frame, or defines it as the site's first epoch
   when the capture lies within 2047 m of it (else it is a new site); a capture with a located
   point joins the nearest framed site whose boundary is within 25 m; anything else is a new site.
   Only framed sites are spatial candidates. The located point is decided once, before the run
   (`worker/params.located_point`): a hand placement in the job's params (the upload page's typed
   place) wins over the capture's own coordinate, and the same point goes to the spatial join,
   to `normalize` and to `georeference`. The phone key may name only a site a phone capture made
   (409 `site_not_phone`): it is a weak shared credential, and naming a seeded or hand-made site
   would fix that site's frame where the phone happens to be. The same rule holds its spatial
   join: a phone-key capture joins by location only a site a phone capture made, so standing in
   a site reaches no more than naming it would; a desktop capture joins any framed site.
5. **The worker passes the frame** as the `site_frame` param (`{siteId, lat, lon, height,
heightDatum, epsg}`) to the stages that write the placement: `ingest_splat`, `manual_placement`,
   `exif_gps`. `georef.json` then _is_ the frame origin, and every later stage (`place_splat`,
   `splat_tiles`, `splat_ground`, `catalog`) gets the frame from it with no second source of truth.
   The placing stage echoes what it used as `georef.siteFrame`.
6. **Exact-origin consistency.** A run of an epoch whose site has a frame must be placed about the
   frame origin exactly, or it is refused (`FrameMismatch`) and the job dead-letters. No tolerance:
   the frame's float8s become JSON (Python's shortest round-trip `repr`) and come back the same way,
   so an honest run matches to the last bit. Backfilled frames come from the same JSON numbers.
7. **Sidecars carry only within an epoch**: from the epoch's current asset, and a new epoch starts
   empty. `LiveMoved` is checked against the epoch's own current asset.
8. **The site's metadata is a deprecated mirror**: `jobId`, `registration`, `coverageUrl` and
   `keepVerifiedPct` are still written on a pipeline-made site for `view.html`, the real-size tool
   and older readers, by runs of the site's first epoch only (the splat those readers show), so a
   later epoch's coverage never lies over epoch 1's tiles; the epoch row is the record of truth,
   and M2b moves those readers onto it. `view.html` (`src/view/shown.ts`) shows the current
   revision of the oldest epoch that has tiles, not the first splat listed (a re-run of epoch 1
   lists its new revision after epoch 2's asset), and lays the mirrored coverage over it only when
   that epoch is epoch 1.
9. **Migration 0032** backfills a frame and epoch 1 for every site a pipeline run registered (a
   `captureId` naming a capture of that site, and a numeric georef), with that site's first splat
   as the epoch's asset unless it has a `valid_to` (a validity window before 0032; on an epoch
   asset it would read as superseded, so such a splat stays legacy). Seeded and catalog sites stay legacy. The downgrade drops only the new
   tables and columns: no asset row and no `source` is touched.

## The one exception: a sidecar attach repoints the asset's URL

`POST /assets/{id}/sidecars` (`services/attach.py`) moves the current asset's `source.url` to a new
generation instead of adding a revision. It is allowed because nothing the URL identifies changes:
an attach adds files (instances, a fill, a collision grid, a rig) beside the very same tiles, and
the generation it leaves stays, immutable, in storage. A revision per attach would instead make
every sidecar a new "run" of the epoch, with its own row and supersession, for tiles that did not
change. The exception is held to that reason, and checked rather than assumed:

- the new generation is listed back before the asset moves, and the attach is refused (409
  `not_attachable`, the asset untouched) unless its tiles' fingerprint (`sidecars.tiles_fingerprint`:
  the tile tree and every tile's size and ETag) is the current generation's; tiles the store cannot
  vouch for (an object without an ETag) are refused the same way;
- every URL the asset is moved off is appended to `assets.url_history` (JSONB, append-only, oldest
  first: `url`, `movedTo`, `replacedAt`, `reason: "sidecar-attach"`), so what the asset pointed at
  on any day can still be found.

A store that recomputed a multipart object's ETag on a server-side copy would make the check refuse
an attach whose tiles are in fact identical. That fails safe (the asset stays put) and the existing
`basedOn` check already depends on copies keeping ETags.

## Two deviations from D3, stated loudly

- **Recentring stays.** D3 says "no per-capture recentring when a frame is given". It is read as
  "no per-capture origin": an unaligned capture is still centred on its own footprint — that is how
  the pipeline decides which point of the splat the capture's coordinate refers to
  (`stages._frame_from_poses`) — and is then moved by the exact WGS84 east/north/up offset of its
  coordinate from the frame origin, folded into the same step so each centre is rounded once. The
  EXIF-aligned branch never recentred; it expresses its fixes about the frame origin instead of
  their median, and carries no offset.
- **`place_splat`, `splat_tiles` and `splat_ground` are not sent the frame.** They already read
  the origin from `georef.json`; a param would be a second source of truth.

And three made while building it:

- A capture registered into a site before 0032 that the backfill could not read (no georef) adopts
  that site's first legacy splat as its epoch's first revision on its next run, instead of adding a
  second visible splat beside it; its sidecars still carry.
- In a frame, ground samples (`splat_ground`) and epoch footprints (`placement.boundary`) are
  placed by the exact inverse of the tiles' root transform (ENU → ECEF → geodetic), not the flat
  111,320 m/° used about a capture's own centre: that rule is about 0.4% long north-south at 28° N,
  which is ~9 m at the 2 km a joined epoch can sit from its origin. Outside a frame both are what
  they were. A runtime scale still rescales footprints linearly in degrees about the origin; its
  error spans only (factor − 1) of the offset.
- Footprint containment (`geometry.contains`) allows 1e-9° (0.1 mm, under the SPZ quantum), so a
  runtime scale undone and redone does not grow a site's boundary by rounding.

## Consequences

- Edits in place respect epochs: `PATCH /assets/{id}` refuses a superseded revision and an epoch
  asset's dates and footprint (409 `epoch_managed`); a runtime scale is about the site's frame and
  is refused for a superseded revision, a frameless site, or a site of more than one published
  epoch; a sidecar attach to a superseded revision is refused; the seed never rewrites an epoch
  asset or the boundary of a site that holds epochs.
- `SiteRead.assets` hides superseded revisions; `GET /assets?siteId=` does not.
- Until the M2b timeline, a new epoch is not `default_visible`, so a two-epoch site opens on epoch
  1; the existing date switcher can show the other.
- A capture located only by EXIF cannot be matched spatially before its run; it joins a site only by
  naming it until M3 moves resolution later. Two first captures of a new field processed at once
  make two sites.
- Vertical disagreement between epochs is hidden by the viewer's per-asset ground clamp until M2b
  removes it and M3 aligns epochs visually.
- An epoch up to 2047 m from its frame's origin, plus its own extent, can pass SPZ's 24-bit range
  in the root tile. Nothing wraps: `pack_spz` refuses such a position (M1a-A's refusal, carried on
  this branch), every fix of an EXIF-aligned epoch is held to the range, and per-tile origins (M1)
  are what will let such an epoch pack.
- The one end-to-end test of the EXIF-aligned branch in a frame
  (`test_exact_gps_in_a_site_frame_recovers_the_scene_in_that_frame`) needs the COLMAP 3.9.1
  binary: it runs in CI only. Locally the branch's frame handling is covered by unit tests.
