# Mission planning: product definition and development record

This is the living record of the plan feature: what it is for, how it was decided, what
shipped, and what is measured. It follows the feature-development workflow used for it:
discovery → definition → design → build in increments → quality gates → launch → review.

## 1. Discovery

**Problem.** Operators of an autonomous land-management fleet plan work in spreadsheets and
chat: which machine, which zone, which days. The console showed plans but could not make one.

**Users and jobs.**

| Who                    | Job to be done                                                  | What they need to see                                     |
| ---------------------- | --------------------------------------------------------------- | --------------------------------------------------------- |
| Fleet operator         | Get tomorrow's work onto the machines without a spreadsheet     | Who goes where and when, on the map; one-sentence changes |
| Land manager           | Know what a season's treatment costs and whether it is on track | Acres, machine-hours, days, with the assumptions visible  |
| Agronomist / ecologist | Treat the right ground at the right time                        | Seed-set windows, follow-up passes, verification imagery  |

**Competitive bar** (planning step only): drone mission planners (DJI FlightHub 2,
DroneDeploy) preview the flight path and estimate time and battery; ag operations centres
(John Deere Operations Center, Trimble) keep work plans per machine with as-applied maps;
robot-fleet consoles queue fields per machine. None plans from a stated goal, none explains
its estimate, none lets the operator change a plan with a sentence.

## 2. Definition (PRD)

**Vision.** A plan is a mission for the fleet. The agent drafts, the map shows, the operator
decides.

**Principles.** The map is the plan · the agent proposes, the operator disposes · every
number has a source · plans are living · ask only when it matters.

**Scope.** Goal → draft → review → approve → dispatch → monitor → revise. Drafting by Claude
through the official SDK with a rule-based fallback that is always labelled. Plans persisted
by the API with a revision history. Conflicts across plans surfaced before approval. Rates
learned from the work log. Execution progress against the schedule.

**Out of scope for now.** Field-ready routing (headlands, obstacles, implement widths from
the robot bridge), weather and daylight windows, streaming drafts, drawing new zones inside
the composer.

**Success metrics.** Time to first approved plan (< 2 min for a new operator); redrafts
before approval (median ≤ 1); plans approved without editing title or objective (> 70%);
estimate error within ±25% once learned rates exist; plan adherence (dispatched → done
without manual replan). The console records the first three per session (dev panel,
"Planning" row) so they can be read during pilots.

## 3. Design

**UX flow.** Plans window → "+ New plan" (or Fleet "Plan a mission", a zone's "Plan here",
or the command bar `plan: …`) → composer (goal, zone and machine chips, examples) →
"Draft with the agent" → review (source note, title, objective, estimates with assumptions,
schedule lanes, steps, what changed, risks, questions, refine box) → Approve → plan detail
(schedule, steps, progress, lifecycle) → Dispatch / Pause / Resume / Revise.

**States.** Composer: idle, drafting (agent stream shows a running thread), ready, error.
Plan: Scheduled → Dispatched → (Paused ⇄) → Done. Revisions increment on every approval of a
revise flow; the detail shows the history.

**Map.** A draft is drawn the moment it exists (zones in scope, coverage passes in the colour
of the machine assigned, numbered markers, dashed route); the camera fits the zones beside
the window. See `docs/MISSION_CONTROL.md`.

**Technical design.**

- API: `POST /api/v1/agent/plan-draft` (planner, stateless) and `/api/v1/plans` (CRUD,
  revisions). Plans are keyed by `project_id` (the mission project, e.g. the demo project
  or a fleet backend's id) and optionally linked to a catalog site. Steps, estimates,
  assumptions, risks and questions are stored as JSON columns: they are the planner's
  output and change with it.
- Web: the mission store merges API plans into the project; the browser store is only the
  offline fallback. Conflict detection and learned rates are pure functions in
  `apps/web/src/missions/` and unit-tested; the same facts are sent to the planner so it
  avoids conflicts rather than just reporting them.
- Quality: rules planner unit tests; a planner evaluation set (goals over the demo project
  with structural expectations) that runs against the rules planner in CI and against
  Claude when a key is present.

## 4. Build record

| Increment | What shipped                                                                               | Commit  |
| --------- | ------------------------------------------------------------------------------------------ | ------- |
| 1         | Goal → draft (Claude / rules), review, approve, agent stream, command bar                  | 1675d23 |
| 2         | Plan on the map, per-machine schedule, assumptions, redraft diff, lifecycle                | 5ca25ba |
| 3         | Plans persisted in the API with revision history; browser store as offline fallback        | _below_ |
| 4         | Conflict checks across plans (machine double-booking, zone overlap), fed to the planner    | _below_ |
| 5         | Learned rates from the work log used in estimates and stated in assumptions                | _below_ |
| 6         | Execution progress against the schedule; replan from drift                                 | _below_ |
| 7         | Planner evaluation set; planning metrics in the dev panel; keyboard and screen-reader pass | _below_ |

## 5. Decisions

- **Structured output over free text.** The draft is a schema the UI renders and the map
  draws; the model never writes UI text that is not also data.
- **Rules fallback, always labelled.** The flow works without a key; a draft never hides
  which planner made it.
- **JSON for planner output in the database.** Steps and estimates are versioned by the
  planner, not by migrations; only identity, status and scope are columns.
- **Conflicts are facts for the planner first, warnings for the operator second.**

## 6. Review

Filled in at the end of the workflow: what the metrics read in the sandbox, what was cut,
what the next cycle takes.
