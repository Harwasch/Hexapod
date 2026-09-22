# ADR 0007 — Fly.io runs the API and the worker; Cloudflare runs the edge

**Status:** accepted · **Date:** 2026-09-22

## Context

Until C1 nothing in the repository said where this runs. `docs/DEPLOYMENT.md` described a
topology — static bundle, container, managed Postgres, object storage — with no `fly.toml`,
no `wrangler.toml` and no deploy workflow behind it, so the topology was a preference
rather than a configuration. C1's job was to commit one.

The original choice was Fly, made on the assumption that "Cloudflare compute" meant
Workers. That assumption was decisive and it is no longer true: the API is a mypy-strict
FastAPI service whose dependencies include `psycopg` and `shapely`, both C extensions,
neither of which runs under Pyodide — Cloudflare Workers would have meant a TypeScript
rewrite of the whole service. **Cloudflare Containers are reported to have reached GA in
April 2026** and run an ordinary Docker image, which removes that objection entirely. So
the choice was genuinely open again, and worth real money and operational simplicity:
Pages, R2, Access and the API would be one bill, one dashboard, one identity boundary,
with R2 reads in-network.

What did not change is the shape of the two processes, which is where the decision turned.

- **The API** is bursty and single-user. It idles for hours and then answers a handful of
  requests. That is exactly what a sleep-on-idle, bill-per-active-CPU-millisecond runtime
  is for, and Containers would suit it better than a machine that is up all month.
- **The worker** is a long-running poll loop (`app/worker/loop.py`). It claims a job,
  commits a lease immediately and renews it every `WORKER_POLL_S` (2 s) for as long as it
  supervises the run — and a Lane 2 `train` stage dispatched to a GPU host is measured in
  hours, during which the worker's whole job is to still be there: poll the dispatched
  stage, sync `checkpoint/` back, and notice a preemption (B1b). A runtime that sleeps
  instances on idle has no place to put that. The Cloudflare-shaped answer is a cron
  trigger that wakes something briefly, which changes the worker from a supervisor into a
  reconciler — a real redesign, of the one component whose lease semantics A7 spent a
  whole step getting right.

There is a second cost that is easy to miss: a Cloudflare Container is fronted by a Durable
Object, so deploying one means writing and maintaining a TypeScript Worker entrypoint that
proxies requests into the container. Not a rewrite of the API, but a new language, a new
build and a new moving part in front of a service whose current deployment artifact is a
Dockerfile that CI already builds and exercises end to end.

And a third: `app/worker/runner.py` has a runner seam — stub, local, cloud — precisely so
that where a stage executes is not baked into the worker. Re-shaping the worker around one
provider's lifecycle is the coupling that seam exists to prevent.

## Decision

Split along the line the workloads already draw.

| What                | Where                        | Why                                                            |
| ------------------- | ---------------------------- | -------------------------------------------------------------- |
| API (`app` process) | Fly.io                       | one image, `fly.toml`, scale-to-zero via `auto_stop_machines`  |
| Worker              | Fly.io, second process group | long-lived, holds a lease, needs a volume for the run workdir  |
| Web bundle          | Cloudflare Pages             | static, three HTML entry points, no server                     |
| Captures and tiles  | Cloudflare R2                | S3-compatible, no egress fee, the browser talks to it directly |
| Postgres + PostGIS  | Neon                         | managed, PostGIS available, scales to zero                     |

So: **Fly for compute, Cloudflare for the edge.** The consolidation argument is real but it
buys a single bill at the price of redesigning the worker around a lifecycle that does not
fit it. The worker is the part of this system that has already been made to survive crashes,
preemption and reclaim; it is the last thing to rebuild for a billing model.

Concretely, `fly.toml` runs both processes from `infra/api.Dockerfile` — the same image CI's
`image` job builds and exercises — as `app` and `worker` process groups, with a 50 GB volume
mounted at `/data` on the worker group only and `WORKER_WORKDIR=/data/worker`.

## Consequences

- **Two providers, two credentials, two dashboards.** `FLY_API_TOKEN` and
  `CLOUDFLARE_API_TOKEN`, both named in `.github/workflows/deploy.yml`.
- **The API is cross-origin to the bucket and to the web app**, so `API_CORS_ORIGINS` and
  the bucket's CORS configuration (`infra/cors/production.json`) are load-bearing rather
  than incidental. They would have been load-bearing on Cloudflare too; being on one
  account would not have made them optional.
- **Cloudflare Access cannot cheaply front the API**, because the API is not behind
  Cloudflare. The write token (A3) remains the only gate on writes, which is what the code
  already assumes.
- **Reversible, and cheaply.** What is provider-specific is `fly.toml`, forty lines that
  name an image, two commands, a volume and a health check. The image is the artifact; a
  move to Cloudflare Containers, Cloud Run or ECS is a new config file against the same
  image. What would not have been reversible is a worker re-shaped around cron wakeups.
- **Revisit when** the worker stops being a supervisor — if every GPU stage becomes a
  webhook-completed dispatch with no polling, the objection above disappears and
  consolidation becomes mostly free.

## What is unverified

Everything about Cloudflare Containers here is second-hand. `developers.cloudflare.com` is
unreachable from the environment this was written in, so the April 2026 GA date, the
sleep/billing behaviour, and the Durable Object entrypoint requirement are **reported, not
checked**. If they are wrong in a way that matters — in particular if Containers now run a
long-lived process without sleeping — this decision deserves re-opening rather than
defending. Nothing has been deployed to either provider: no `fly deploy`, no
`wrangler pages deploy`, no bucket created, no account that exists.
