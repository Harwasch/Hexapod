# Mission control

The web app is the operator console for autonomous land-management robots. The map is the
hero: one CesiumJS world from Earth down to the machines working a site, with the mission
layer (zones, machines, plans, the agent's activity) drawn over it. This document describes
that layer and the seam where real fleet data plugs in.

## Layout

```text
┌ Project badge ──── Search ───────────────── Map · Plan · Fleet ┐
│ (site picker)                                Imagery Vegetation Zones Tracks
│ Tool rail        (machine markers, zone chips over the world)   Inspector / Dev
│ ┌ panels ┐                                                      ┌ Plans / Fleet window ┐
│ Feeds · Selection card                                          Agent activity stream
└ Command bar ── "select TR-07" · "show zones" · "fly to …" ── readouts ┘
```

- **Project badge** (`features/mission/ProjectCard`) names the active site's project; its menu
  lists catalog sites and opens Add Data.
- **View tabs** (`ViewTabs`) switch `map` / `plan` / `fleet`. Keys `1` `2` `3`.
- **Layer pills** (`LayerPills`) toggle imagery, vegetation, zones and tracks.
- **Markers and chips** (`MissionOverlays`) are DOM elements positioned every frame from
  `MissionManager` anchors, so they stay crisp and accessible (real buttons, keyboard focus).
- **Selection card** shows a machine (battery, acres, shift, Pause / Camera) or a zone
  (progress, Open plan / Reassign).
- **Plans** and **Fleet** windows list plans with detail + "Show on map", and machines with a
  treatment log.
- **Agent stream** (`AgentStream`, key `a`) is the running log of agent actions and command
  replies.
- **Command bar** (`CommandBar`) accepts short intents; `lib/intents.ts` parses them locally
  (machine and zone ids, views, layers, measure, camera, site fly-to, then geocoding).

## Data flow

```text
catalog site ──▶ SceneBridge ──▶ MissionProvider.projectForSite() ──▶ useMission (Zustand)
                                                   │
                                                   └──▶ scene.mission (MissionManager)
                                                        zones as clamped polygons
                                                        machine tracks as polylines
                                                        per-frame anchors → overlays
```

`apps/web/src/missions/types.ts` defines `Project`, `Machine`, `Zone`, `Plan`, `AgentAction`,
`WorkLogRow` and `Feed`. `MissionProvider` is the only seam: it returns a `Project` for a site
or `null`.

## Demo data is simulated

`missions/demo.ts` ships one project, **Blackrock Mesa**, attached to the public Cesium splat
site. Every record carries `simulated: true`, the project badge says so, and nothing in the
demo is presented as a live feed (camera feeds show "no stream"). Replace
`DemoMissionProvider` with a provider backed by your fleet API (REST/WebSocket/MCAP) to get
live positions; the overlays, cards and command bar do not change.

## Design language

The look is the "Robot Land Management UI" design: near-black olive base, dark glass panels
(`packages/ui/src/styles/tokens.css`), teal for working/done, blue for the agent, amber for
attention, Instrument Sans for text and Azeret Mono for readouts. Fonts load from Google
Fonts with system fallbacks; the light glass theme is opt-in in Settings.

## Tests

- `src/__tests__/intents.test.ts` — command parsing.
- `e2e/app.spec.ts` "mission control" — tabs, plan → show on map, fleet → selection,
  command bar replies, layer pills.
