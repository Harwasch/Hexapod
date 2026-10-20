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
   current revision, `observed_at` (with where it came from: the capture, the run's manifest, or
   the upload time, labelled `created`), its footprint, platform (`phone-video` or `unknown` until
   M3), status (`published`; `rejected` and `superseded` are reserved for M3's gates), how it found
   its site (`explicit`, `spatial`, `new-site`), alignment (M3) and quality. One capture is one
   epoch; `captures.epoch_id`, for several captures of one day, is future work.
3. **An asset is one tileset of one epoch and one source kind** (`assets.epoch_id`,
   `assets.source_kind`: `observed`, or `inferred` from M5). A re-run of the same capture inserts a
   new asset revision of its epoch and supersedes the old one by setting its `valid_to` (and
   `default_visible = false`) — the only two writes ever made to it. Its URL is never rewritten.
   The epoch's current asset is its newest observed asset with no `valid_to`. A republish onto the
   URL the epoch already shows is no revision: generations are content-hashed.
4. **The site is resolved when the job starts**, because the pipeline has to place the capture in
   the frame while it runs (`services/epochs.frame_for_run`): a re-run keeps its epoch's frame; a
   capture that names a site (`siteId` on capture create and on the phone route, kept in
   `captures.metadata.siteId`) uses that site's frame, or defines it as the site's first epoch; a
   capture with a coordinate joins the nearest framed site whose boundary is within 25 m; anything
   else is a new site. Only framed sites are spatial candidates.
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
   and older readers; the epoch row is the record of truth, and M2b moves those readers onto it.
9. **Migration 0032** backfills a frame and epoch 1 for every site a pipeline run registered (a
   `captureId` naming a capture of that site, and a numeric georef), with that site's first splat
   as the epoch's asset. Seeded and catalog sites stay legacy. The downgrade drops only the new
   tables and columns: no asset row and no `source` is touched.

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
- The ground-sample check in the pipeline tests uses 2e-6° rather than 1e-7°: the ground samples
  convert metres to degrees with a flat 111,320 m/°, about 0.4% long at 28° N.
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
- A site several hundred metres across in one tile can still wrap SPZ positions until M1's refusal
  to wrap is merged; M2a merges after it.
