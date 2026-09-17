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
  treatment log. "+ New plan", "Plan a mission" (Fleet) and "Plan here" (a zone without a
  plan) open the plan composer described below.
- **Agent stream** (`AgentStream`, key `a`) is the running log of agent actions and command
  replies. It has no card while the agent is idle: it appears when something runs (a plan
  being drafted, a demo thread), when a command was answered, or when opened with `a`.
- **Command bar** (`CommandBar`) accepts short intents; `lib/intents.ts` parses them locally
  (machine and zone ids, views, layers, measure, camera, site fly-to, then geocoding).

## Planning with the agent

A plan is a mission for the fleet, and the agent drafts it. The flow is goal → draft → review
→ approve:

1. **Goal.** The operator writes what the fleet should achieve ("clear the star thistle from
   Z-14 and Z-21 with two mowers before seed set") in the plan composer, or in the command bar
   as `plan: …` / "draft a plan to …". Zones and machines can be pre-selected with chips; a
   selected zone or machine seeds them.
2. **Draft.** The console sends the goal plus the project's zones, machines and existing plans
   to `POST /api/v1/agent/plan-draft`. The API drafts with Claude (official SDK, structured
   output, model from `ANTHROPIC_MODEL`, default `claude-opus-5`) when `ANTHROPIC_API_KEY`
   is set; otherwise a rule-based planner drafts and every draft says so (`source`, `note`).
   The key never reaches the browser. While drafting, the agent stream shows the running
   thread.
3. **Review.** The draft shows title, objective, zones, machines, cadence and dates, the
   estimate (acres, machine-hours, days), ordered steps, risks and the agent's questions.
   Title and objective are editable; "Redraft" sends an answer or adjustment ("use three
   machines", "skip Z-21") together with the previous draft; "Show on map" flies to the zones.
4. **Approve.** The plan joins the Plans list as _Scheduled_ with its steps and provenance
   (Claude + model, or rule-based). "Revise with agent" reopens the composer with the plan's
   goal and replaces it on approval.

Approved plans live in the browser (`twin.mission.v1`, per project) until a fleet backend
persists them; the `MissionProvider` seam is unchanged. `apps/web/src/missions/planDraft.ts`
converts between the API draft and the console's `Plan`; `apps/api/app/services/planner.py`
holds both planners behind one `Planner` protocol and is covered by `tests/test_agent.py`.

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
