# twin-api

FastAPI + PostGIS catalog service for the digital twin. See the repository README for the full quickstart and `docs/DATA_MODEL.md` for the schema.

```bash
uv sync
uv run alembic upgrade head
uv run python -m app.seed
uv run uvicorn app.main:app --reload --port 8000
```
