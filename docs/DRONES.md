# Drone missions: before first flight

There is no drone yet. Test captures are walked or filmed with an iPhone. So the drone side
of mission planning is a **stub**, behind a build-time flag that is **off by default**:

- `VITE_ENABLE_DRONE_EXPORT`, read in `apps/web/src/app/env.ts` as `env.droneExportEnabled`.
  It is a public `VITE_` boolean, never a secret.
- **Off** (every build today): the scout card offers no DJI route download, no height picker
  and no battery figures; it says "Drone missions: not available yet". A re-scan of weak
  areas asks for no battery budget, and its plan says to walk or film the areas with a phone.
- **On**: the DJI KMZ export (`apps/web/src/missions/wpml.ts`, `ScoutExport.tsx`) and the
  drone wording come back. Do not turn it on for a real flight until every item on this page
  is closed.

Most items come from a Codex review of the export (PRs #68 and #69, 2026-10-20).

## 1. Choose the drone and the app

Nothing is chosen. The code assumes a **DJI Mavic 3 Enterprise (M3E) wide camera, flown from
current DJI Pilot 2** (drone enum 77/0, payload 66). That is an **unconfirmed assumption**.

- Pick the aircraft, its camera, and the app that flies the route (DJI Pilot 2 or FlightHub
  2), with their firmware versions.
- Update `SCOUT_CAMERAS` in `apps/web/src/missions/scout.ts`, and the enum values the KMZ
  writes, if the model differs.

## 2. WPML fields to check against DJI's spec

The writer emits these, but the review could **not find them in DJI's WPML documentation**
(the template, waylines and common-element tables). Each needs a DJI source, or must go:

| Field                        | Where the writer puts it                       | Open question                                      |
| ---------------------------- | ---------------------------------------------- | -------------------------------------------------- |
| `distance`                   | `waylines.wpml`, the wayline's `Folder`        | Not in DJI's tables.                               |
| `duration`                   | `waylines.wpml`, the wayline's `Folder`        | Not in DJI's tables.                               |
| `payloadSubEnumValue`        | `missionConfig` → `payloadInfo`                | Not in DJI's tables.                               |
| `waypointHeadingAngleEnable` | `waylines.wpml`, each waypoint's heading       | Not in DJI's tables.                               |
| `waypointHeadingPoiIndex`    | `template.kml`, the global heading             | Not in DJI's tables.                               |
| `caliFlightEnable`           | `template.kml`, the waypoint template `Folder` | Documented only for M300/M350, in other templates. |
| `globalRTHHeight`            | `template.kml`, `missionConfig`                | Its placement in the template is unverified.       |
| `payloadLensIndex`           | the `takePhoto` action                         | Support on the M3E is unverified.                  |

These **required** fields are **missing**. DJI marks both required; its examples do not
always match its tables, so this is a conformance gap, not a proven import failure:

- `gimbalHeadingYawBase` in the `gimbalRotate` action.
- `waypointHeadingPathMode` in the template's global heading parameters.

**`ellipsoidHeight`** is set to the relative height (metres above take-off). An ellipsoid
height is measured from the WGS84 ellipsoid, so at a take-off 500 m up a 60 m route must not
say 60 m. `useGlobalHeight=1` limits the effect today. Omit it, or convert it properly.

## 3. Route shape

- **One wayline per KMZ?** The export writes several waylines into one KMZ: an oblique's two
  crossing passes are two, an orbit (two laps) is one, and consecutive nadir areas share one.
  Keep to one wayline per KMZ unless DJI's docs explicitly allow several, and cite the page
  if they do.
- **Waypoint-count limit.** Nothing caps the waypoints. DJI's documented index range ends at
  65,535, and a very large area produced about 100,000 in the review. Find the practical
  limit for the chosen aircraft and app, and refuse (or split) above it.
- **Battery budget.** A re-scan's patches are budgeted for one battery of the drone class's
  60 m nadir lanes, but a patch may fly as low as 20 m, so "within one battery" can be false.
  Size it by each patch's height and pattern before relying on it.
- **Obliques on small patches.** Crossing obliques are flown as nadir lanes with the camera
  pitched forward. On a small patch the camera looks past it: at 60 m and −30° the nearest
  ground in view is about 35 m ahead. Project the camera footprint, or refuse patches too
  small for the pitch.
- **Repeated points.** An orbit can export two identical consecutive points after rounding.
  Deduplicate after rounding, and check the aircraft's minimum waypoint spacing.

## 4. Where the drone may fly

- **The site boundary and its holes are the flight limits.** No flight path, transit leg or
  orbit may leave the boundary or enter a hole: holes are no-fly, and so are holes in an
  area. A zone that cannot be flown inside those limits is refused, with the reason.
  Enforced by `flightRefusal` (`apps/web/src/missions/flightLimits.ts`): it samples every leg
  at most a metre apart (on routes up to 200 km), with a metre's tolerance at an outline. The limit is the Land's boundary
  zone, else the site project's `SITE` zone; a project with neither (the demo ranch's own
  fields) is offered nothing. It refuses; it does not re-route round a hole. Separately
  stored Land exclusions that are not holes are not checked yet.
- **No download for an empty or partial route.** `routeRefusal` names each area with no
  lane, no orbit, or only one of its two crossing passes, and nothing is offered to fly.
- Heights are relative to take-off and do not follow terrain. The relief check assumes take-off
  at the first waypoint and only warns. Transit, take-off and return paths are not checked for
  obstacles.

## 5. Tests still to write

- [ ] A test that **unzips the KMZ** and validates its structure: `wpmz/template.kml` and
      `wpmz/waylines.wpml` present, both parse, the namespaces right, every wayline with its
      waypoints and action groups. Today the tests parse the XML before it is zipped and check
      only the ZIP's file names and trailer.

## 6. Before any flight

1. **Test import into DJI Pilot 2** (and FlightHub 2, if used) on the chosen aircraft's
   controller: the route loads with no warning, and after an import and save every template,
   wayline, pitch, point of interest and relative height is unchanged.
2. **Ground check**: fly nothing yet. Walk the route on the map and on site; confirm the
   take-off point, the return-home height, the boundary and holes, and the airspace (B4UFLY or
   LAANC, Part 107).
3. Only then a first supervised flight, at a safe height, over open ground.
