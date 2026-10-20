# ADR 0014 — Ingest metadata and platform: every kept frame is described from its original

**Status:** accepted · **Date:** 2026-10-19 · **Decision record:** D4 and D6 of the splat-pipeline
build (M3a part 1)

## Context

`normalize` shrinks a full-size photo to the size training reads and keeps only its EXIF block.
Everything else a camera wrote — above all a DJI drone's XMP packet, with its RTK fix, gimbal and
flight angles and per-unit calibrated focal — was gone before `pose` or `georeference` saw a frame.
The epoch's platform (D4's survey tier) was a guess from the upload route (a phone key is a phone
video; everything else `unknown`). A phone's own location accuracy was stored and never read. An
iPhone's HDR (HLG) video was cut into frames as if it were BT.709. M3b's registration needs the
fixes, their accuracy and the tier; this ADR records what M3a reads and how far it can be trusted.

## Decision

1. **`frames_meta.json`** (`frame_meta.py`, produced by `normalize`): one record per kept frame,
   keyed by the renumbered frame name, read from the original. Shape `frames_meta/1`: `name`,
   `original`, `originalPath`, `originalSize`, `frameSize`, `candidateIndex` and `offsetS` (video),
   `time` `{local, offset, utc, source}`, `gps` `{lat, lon, altitudeM, altitudeDatum, altitudeNote,
hAccuracyM, dop}`, `camera`, `exposure`, `focal` `{mm, mm35, px, pxFrom}`, `dji`, `mask`,
   `unreadable`. Top level: `platform`, `focal` (the capture's one focal, or none when frames
   disagree by more than 0.5%), `timeRange`, `exposure`, `masks` (COLMAP's `<name>.png` convention,
   produced by part 2). Masks are named, not written here: an artifact has one producer.
2. **Times are UTC only when a zone is known**: EXIF `OffsetTimeOriginal`, else the GPS date and
   time, else local only (`utc: null`), else the XMP `CreateDate`. A video frame is the clip's start
   (QuickTime creation date, assumed to be the start of recording, unverified; else the mvhd time)
   plus `candidateIndex / rate`. The earliest becomes `source_meta.capturedAtUtc`, which dates the
   epoch first (ADR 0013) and is **not** copied into `captures.captured_at`, which stays what a
   person said.
3. **Heights use the site frame's datum vocabulary**: an EXIF fix's `altitudeDatum` is
   `msl-unverified`, with the words in `altitudeNote`. DJI's `AbsoluteAltitude` datum is
   `dji.absoluteAltitudeDatum: "unverified"`, deliberately outside the site-frame datums so it never
   compares equal to one, with the forum claim (ellipsoidal when RTK-fixed) in `absoluteAltitudeNote`.
4. **DJI field status.** Checked against ExifTool's DJI table: `RtkFlag`, `RtkStdLon/Lat/Hgt`,
   `CalibratedFocalLength`, `CalibratedOpticalCenterX/Y`, `Gimbal*`, `Flight*`,
   `Absolute/RelativeAltitude`, `GpsLatitude/GpsLongitude` and the misspelt `GpsLongtitude`,
   `DewarpFlag`. Not confirmed by a primary source, and named in each record's `unverifiedFields`:
   `RtkDiffAge`, `GpsStatus`, `AltitudeType`, `UTCAtExposure`. RTK flag codes (0 none, 16 single,
   34–49 float, 50 fixed) are Sitemark's. The calibrated focal is taken as pixels of the original
   (DJI's P4 RTK documentation); unverified for the M3E. **No real Mavic 3 Enterprise file has been
   read**: the fixtures are synthetic. Serial numbers are never read.
5. **Platform** (`capture_platform.py`), first rule that matches: a requested `platform`; every
   frame RTK-fixed (`RtkFlag` 50) with a stated accuracy inside one GPS L1 wavelength (0.19 m)
   → `rtk-drone`; at least half with DJI flight XMP → `drone`; a DJI video
   make → `drone` (unverified, no sample); Apple QuickTime or Android keys → `phone-video`; a phone
   make on at least half the photos → `phone-photo`; a phone-key upload → by format; else
   `unknown`. "At least half" is a classification rule, not a tolerance. The value goes
   `source_meta` → manifest → `registration.json` → **`epochs.platform`**, on the first run and every
   re-run; a run that detects nothing or `unknown` keeps a known value. **No `captures.platform`
   column and no migration**: a mapped capture column breaks the 0007-revision ORM tests, and the
   epoch is where D4 puts the tier. **A platform label never makes a capture surveyed.** Each
   platform carries `platformEvidence`: `metadata` (the frames' own XMP, EXIF or QuickTime keys
   showed it), `claimed` (only a request — a capture's `metadata.platform`, which any write-token
   client may set — or the phone-key route said so) or `none` (`unknown`). A request the frames
   bear out is `metadata`. It travels with the value to `registration.json`, and the worker writes
   `rtk-drone` onto the epoch only with `metadata` evidence; a claimed one is recorded as `drone`.
6. **HDR**: an HLG or PQ clip is tonemapped before frames are taken (`zscale` linearise at
   `npl=100`, BT.709 primaries, Hable, BT.709 transfer and matrix), in extraction and in
   `max_side: auto`'s sampling. `npl=100` is untuned (BT.2408's 203 nits is the alternative). A
   binary without the filters takes frames as they are and says why.
7. **Exposure**: luma mean and clipped percentages per kept frame, EV100 per photo, flags (clipped
   frames, EV swing, brightness swing) at first-principles thresholds (5%, 1 EV, 1 stop) that are
   information only. The fail gate (`max_clipped_frames_fraction`) ships **off**: no measurement
   backs a threshold. `QualityGateError(gate, measured, limit, *, unit, reason)`,
   `parse_quality_gate` and `check_gate` are defined here with M3b's signature; the error is in
   `BAD_INPUT_ERRORS` and never retried. An EXIF setting no camera can record (outside f/0.5–f/1024,
   1 µs–1 day, ISO 1–10,000,000; `frame_meta.EXPOSURE_RANGES` gives the physics) is dropped and
   named in the frame's `implausible` list, and its EV100 is null: a value in a file never fails
   the run.
8. **Focal**: `pose` `focal_from: exif` **seeds** COLMAP's camera from the capture focal and lets
   bundle adjustment refine it (`hold_focal: true` holds it). A 35 mm focal converts by the diagonal
   with COLMAP's own 43.27 mm constant (pycolmap 4.2 reads 443.7254 px for 24 mm at 640×480).
   `poses.json` records `initialPx`, `initialFrom`, `held` and `recoveredOverInitial`, the ratio
   that will decide, across captures, whether holding is ever right. A focal is used only when it
   gives a field of view across the long side of 0.1°–179° (`frame_meta.FOCAL_FIELD_OF_VIEW_DEG`: a
   pinhole camera sees less than 180°); one outside is dropped and named in the frame's
   `implausible` list, the next source is tried, and `pose` checks the seed again before using it.
9. **Phone accuracy** (`metadata.locationAccuracyM`, and a video's QuickTime accuracy) widens the
   georeference's `uncertaintyM`, never narrows it below a dropped pin's 10 m.

`photo-reconstruct` is version 12 (D6's stack: M1b is 11).

## Consequences

- M3b reads `frames_meta.json` (`gps`, `hAccuracyM`, `dji.rtk.*`, `altitudeDatum`) for its pose
  priors, and extends `QualityGateError` rather than redefining it.
- Until the Modal CPU image is rebuilt, a remote `pose` ignores `focal_from` and its `poses.json`
  lacks `focal.initialFrom`: that absence is how a stale image shows.
- A real M3E photo is the cheapest verification of item 4; until then every unverified field is
  named in the data it appears in.
