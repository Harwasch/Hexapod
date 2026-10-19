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
- CI (`.github/workflows/ci.yml`) runs all of the above per changed area; Deploy (`deploy.yml`) ships main.

## Conventions

- Layout: `apps/web` (React + Vite + CesiumJS), `apps/api` (FastAPI + PostGIS), `packages/{contracts,geo,world,ui,config}`,
  `tools/{captures,pipeline}` (Python capture and splat pipeline), `workers/worlds`, `infra/`, `docs/` (ADRs in
  `docs/DECISIONS/`).
- One step per commit; the message explains the decision, not the diff.
- Run the gate yourself before believing a report; local green has been wrong before, so CI is the judge.
- Never calibrate a tolerance on one machine's arithmetic. What cannot be verified stays stubbed, with the reason.
- Simulated things are labelled as simulated in the UI (fleet, Living Survey motion).
- Trip-wires (`docs/HANDOFF.md` § 6): `data/tiles/synthetic-tree` is byte-identity gated; `openapi.json` is
  committed; `pnpm format:check` covers YAML and Markdown and runs before lint; an e2e test asserts the exact
  tool-rail button count; the repo's `.env` leaks into `apps/api` settings tests, so override explicitly.
- `VITE_` variables are public. Server secrets never carry that prefix.
- Past experiments and parked work: `docs/ARCHIVE.md`; check it before starting research or a bake-off.

## Branches and PRs

Integration branch: `main`. One short-lived branch per task, `<engine>/<type>-<slug>` (`claude/feat-fog-v2`,
`codex/fix-watcher-replays`), types `feat fix clean hw sim research review docs`. Draft PR against `main`;
squash-merged after review. Never commit straight to `main`. Bug fixes start with a failing test.

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
