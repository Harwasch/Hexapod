# Hexapod monorepo

The **Living World** digital twin (a time-aware CesiumJS globe down to centimetre scans) with **mission control**
for a simulated land-management robot fleet (demo: Oak Bench Ranch), plus the hexapod's hardware design work.
Deployed at https://twin-web-f57.pages.dev (Cloudflare Pages + R2; API and worker on Fly; Neon PostGIS).
Start with `README.md`, then `docs/HANDOFF.md` and `docs/DEPLOYMENT.md`.

## Build and test

- Setup: Node 22.12+, pnpm 10, Python 3.12, `uv`, Docker. `pnpm install`; `cd apps/api && uv sync`.
  PostGIS + MinIO: `docker compose -f infra/docker-compose.yml up -d`.
- Test (the command briefs name in DONE WHEN): `pnpm format:check && pnpm lint && pnpm typecheck && pnpm test`
  - plus `pnpm e2e` when `apps/web` changed (Playwright; `pnpm --filter @twin/web exec playwright install chromium`)
  - plus `uv run ruff check . && uv run ruff format --check . && uv run mypy . && uv run pytest -q` in each Python
    project touched: `apps/api` (needs `TEST_DATABASE_URL`, run `uv run alembic upgrade head` first),
    `tools/pipeline`, `tools/captures`.
- Build: `pnpm build` (web bundle in `apps/web/dist`), then `node apps/web/scripts/check-bundle.mjs`.
- Contracts: `pnpm contracts:check`. After a Pydantic schema change:
  `cd apps/api && uv run python -m app.scripts.export_openapi ../../packages/contracts/openapi.json`, then
  `pnpm contracts:generate`.
- CI (`.github/workflows/ci.yml`): a PR, and a push to main, run only the jobs whose paths changed
  (`.github/scripts/ci_changes.py`, tested by `python3 -m unittest discover -s .github/scripts`); a change to
  CI itself runs everything; a nightly runs everything when main moved. A draft PR runs only the format and
  workflows checks; marking it ready for review (or a `workflow_dispatch` on its branch) runs the full suite.
  Deploy (`deploy.yml`) ships main.

## Conventions

- Layout: `apps/web` (React + Vite + CesiumJS), `apps/api` (FastAPI + PostGIS), `packages/{contracts,geo,world,ui,config}`,
  `tools/{captures,pipeline}` (Python capture and splat pipeline), `workers/worlds`, `infra/`, `docs/` (ADRs in
  `docs/DECISIONS/`).
- One step per commit; the message explains the decision, not the diff.
- Run the DONE WHEN gate yourself after your last edit and report its tail. That local run is the merge gate.
- Never calibrate a tolerance on one machine's arithmetic. What cannot be verified stays stubbed, with the reason.
- Simulated things are labelled as simulated in the UI (fleet, Living Survey motion).
- Trip-wires (`docs/HANDOFF.md` § 6): `data/tiles/synthetic-tree` is byte-identity gated; `openapi.json` is
  committed; `pnpm format:check` covers YAML and Markdown and runs before lint; an e2e test asserts the exact
  tool-rail button count; the repo's `.env` leaks into `apps/api` settings tests, so override explicitly.
- `VITE_` variables are public. Server secrets never carry that prefix.
- Past experiments and parked work: `docs/ARCHIVE.md`; check it before starting research or a bake-off.
- Drone missions are a stub behind `VITE_ENABLE_DRONE_EXPORT` (off): read `docs/DRONES.md` before turning it on.

## Branches and PRs

Integration branch: `main`. One short-lived branch per task, `<engine>/<type>-<slug>` (`claude/feat-fog-v2`,
`codex/fix-watcher-replays`), types `feat fix clean hw sim research review docs`. Draft PR against `main`;
squash-merged after review. Never commit straight to `main`. Bug fixes start with a failing test.
Merge when the last local gate run passed and the brief's review is done; CI is advisory.
A check already red at merge time still blocks merging. Fix a red run on main forward with a new PR.

## Code Review Rules

<!-- read by review bots (Codex code review); repo-wide rules here, service-specific ones in nested AGENTS.md -->

- Flag any change without a test for the behaviour it changes.
- Flag secrets, tokens or credentials in code, config or logs, and any server secret given a `VITE_` prefix.
- Flag a Pydantic schema change without a regenerated `packages/contracts/openapi.json` and TS types.
- Flag CesiumJS imports in `view.html`, `admin.html` or `upload.html` entry points (they must not load Cesium).
- Flag simulated data shown in the UI without a Simulated label.
- Flag tolerances widened without a recorded measurement.
- Leave formatting and lint to CI.

## Lessons

<!-- one line per lesson learned after a merge; repo-specific only -->

- Playwright reuses a dev server already on its port (`reuseExistingServer`), so on a shared machine `pnpm e2e` can
  silently test another worktree's code on 5173. Give every e2e run its own `E2E_PORT`.
- `/tmp` on the shared dev machine is a small RAM disk with a per-user quota. Set `TMPDIR` to a folder under
  `~/.cache` for pytest and Playwright runs: test databases and browsers fill `/tmp`, and the capture worker
  refuses jobs below 5 GB free.
- Heavy jobs run only via `heavy <command>`, one per project at a time, at most 10 cores, 20 GB and about an hour on
  HGeekom; bigger runs, long runs and any GPU or splat training go to the cloud (Modal).
