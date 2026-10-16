# twin-api

FastAPI + PostGIS catalog service for the digital twin. See the repository README for the full quickstart and `docs/DATA_MODEL.md` for the schema.

```bash
uv sync
uv run alembic upgrade head
uv run python -m app.seed
uv run uvicorn app.main:app --reload --port 8000
uv run python -m app.research   # land research worker, a separate process (see below)
```

Land research runs (overviews, terrain, land cover, vegetation, archives, ecology, solar and
AI investigations) are queued by the API and run by `python -m app.research`, which the API
never starts for you. The worker checks in every 15 s; `GET /api/v1/research/status` reports
it as online, stale, stopped or never seen, and a queued run waits until one runs. See
`docs/LAND_EXPLORATION.md` ("Research worker presence").

Optional: `uv run python -m app.seed.scans` also registers the hosted catalog's phone scans
(the camp, the pumpkin and the spool) from the committed snapshot `app/seed/hosted_scans.json`,
so a local catalog has scans with objects and image embeddings. It is idempotent and reads no
network; `--refresh-snapshot` re-reads the snapshot from the hosted API.
