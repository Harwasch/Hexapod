# twin-api

FastAPI + PostGIS catalog service for the digital twin. See the repository README for the full quickstart and `docs/DATA_MODEL.md` for the schema.

```bash
uv sync
uv run alembic upgrade head
uv run python -m app.seed
uv run uvicorn app.main:app --reload --port 8000
```

Optional: `uv run python -m app.seed.scans` also registers the hosted catalog's phone scans
(the camp, the pumpkin and the spool) from the committed snapshot `app/seed/hosted_scans.json`,
so a local catalog has scans with objects and image embeddings. It is idempotent and reads no
network; `--refresh-snapshot` re-reads the snapshot from the hosted API.
