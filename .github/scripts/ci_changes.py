"""Which CI jobs a change needs: the `changes` job of .github/workflows/ci.yml.

Each job in ci.yml is gated on one output of this script. The decision is made here
rather than inline in the workflow so it can be tested (test_ci_changes.py, run by the
workflows job) and run by hand:

    git diff --name-only origin/main HEAD | python3 .github/scripts/ci_changes.py --event push

By event:

- pull_request: every job runs, as before; only the Playwright shards are skipped when
  nothing the web app is built or tested from changed (and only the page specs run when
  the 3D scene's inputs did not change).
- push to main: only the jobs whose inputs changed between the push's `before` and its
  head. A new branch (an all-zeros `before`), a force-push, or a diff that cannot be
  computed runs everything.
- schedule (the nightly): everything, unless main's head is the commit the last nightly
  already tested -- then nothing.
- workflow_dispatch, or anything else: everything.

The mapping is deliberately coarse and errs towards running: a path is matched against
every job's inputs (one file can need several jobs), on a push a path no rule knows about
runs everything, and only `INERT` paths are allowed to run nothing. The nightly is the
backstop for a dependency the mapping misses.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from collections.abc import Iterable, Sequence

# The jobs, by output name. `e2e` gates the Playwright shards; `scene` says whether they
# include the 3D scene specs (six shards) or the app's page specs alone (four).
JOBS = (
    "web",
    "e2e",
    "api",
    "image",
    "captures",
    "pipeline",
    "trainer",
    "worlds",
    "arm",
    "workflows",
)

# What `pnpm install` resolves from: a change here can change any node job.
NODE_INSTALL = r"package\.json$|pnpm-lock\.yaml$|pnpm-workspace\.yaml$|\.npmrc$|patches/"

# JSON the web app imports out of apps/api (apps/web/src/missions/*.ts).
API_JSON_IN_WEB = r"apps/api/app/.*\.json$"

INPUTS: dict[str, str] = {
    # Web · lint, typecheck, test, build. Prettier and ESLint run over the whole repo, so
    # their configs count, and so does any JS or TS outside apps/web (functions/, the
    # worlds worker's browser transport test). The unit tests read data/tiles and the
    # captures' category list.
    "web": "|".join(
        [
            r"apps/web/",
            r"packages/",
            r"data/",
            r"tools/captures/data/",
            r"functions/",
            r"infra/pages/",
            r"wrangler\.toml$",
            r"eslint\.config\.js$",
            r"\.prettierrc",
            r"\.prettierignore$",
            r"\.editorconfig$",
            r"tsconfig[^/]*\.json$",
            r".*\.(js|mjs|cjs|ts|tsx)$",
            API_JSON_IN_WEB,
            NODE_INSTALL,
        ]
    ),
    # Web · Playwright: what the app is built and served from, and the fixtures the specs
    # load off disk. Not ci.yml: a workflow change runs the workflows check only.
    "e2e": "|".join(
        [
            r"apps/web/",
            r"packages/",
            r"data/tiles/",
            r"tools/captures/data/",
            API_JSON_IN_WEB,
            NODE_INSTALL,
        ]
    ),
    # API · lint, typecheck, test, migrations. It imports tools/pipeline and
    # tools/captures, checks packages/contracts/openapi.json, and its tests read
    # packages/geo fixtures, packages/world, data/tiles, infra/ and two web sources.
    "api": "|".join(
        [
            r"apps/api/",
            r"tools/(pipeline|captures)/",
            r"packages/(contracts|geo|world)/",
            r"apps/web/src/(missions|upload)/",
            r"infra/(modal|cors|postgis)/",
            r"infra/docker-compose\.yml$",
            r"data/tiles/",
            r"\.env\.example$",
        ]
    ),
    # Image · the real Dockerfile, which copies apps/api and both tools projects.
    "image": "|".join(
        [
            r"apps/api/",
            r"tools/(pipeline|captures)/",
            r"infra/(api\.Dockerfile$|modal/|cors/)",
            r"fly\.toml$",
            r"\.dockerignore$",
        ]
    ),
    # Captures · its tests read data/tiles, infra/modal apps and the API's sidecars, and
    # its fixture checksum must agree with packages/world.
    "captures": "|".join(
        [
            r"tools/captures/",
            r"data/tiles/",
            r"infra/modal/",
            r"packages/world/",
            r"apps/api/app/services/sidecars\.py$",
        ]
    ),
    # Pipeline · imports tools/captures, builds infra/modal/app.py, renders data/tiles.
    "pipeline": r"tools/(pipeline|captures)/|infra/modal/|data/tiles/",
    # Trainer · infra/modal's requirements and check, against tools/pipeline's argv.
    "trainer": r"tools/pipeline/|infra/modal/",
    # Worlds · the worker, scripts/ (its tests and the capabilities sync), and the web
    # files those read.
    "worlds": r"workers/worlds/|scripts/|apps/web/src/worlds/|apps/web/package\.json$",
    "arm": r"hardware/arm/",
    # Workflows · parse every workflow, and check their secrets against the handover.
    "workflows": r"\.github/|docs/DEPLOYMENT\.md$",
}

# Paths no job reads: they may run nothing. Everything else that matches no rule above
# runs everything, so a new top-level directory or root file is never silently untested.
INERT = r"docs/|[^/]*\.md$|\.claude/|\.agents/|\.gitignore$|skills-lock\.json$|LICENSE$"

# The 3D scene specs (the `scene` and `webgpu` projects, apps/web/playwright.config.ts)
# render from these. A diff that reaches the e2e job but none of these runs the
# `chromium` project alone -- on a pull request. Main runs the scene specs whenever it
# runs Playwright at all.
SCENE = "|".join(
    [
        r"apps/web/src/(cesium|dev|lib|state)/",
        r"apps/web/(public/|[^/]*\.html$|package\.json$|vite\.config\.ts$|playwright\.config\.ts$|tsconfig[^/]*\.json$)",
        r"packages/(world|geo)/",
        r"data/",
        r"tools/captures/data/",
        API_JSON_IN_WEB,
        NODE_INSTALL,
    ]
)
E2E_SPEC = r"apps/web/e2e/"
PAGE_SPECS = r"apps/web/e2e/(admin|app|captures|layout|phoneUpload|view)\.spec\.ts$"

ZERO_SHA = "0" * 40


def _match(pattern: str, path: str) -> bool:
    return re.match(f"(?:{pattern})", path) is not None


def everything() -> dict[str, bool]:
    return {**{job: True for job in JOBS}, "scene": True}


def nothing() -> dict[str, bool]:
    return {**{job: False for job in JOBS}, "scene": False}


def classify(paths: Iterable[str]) -> tuple[dict[str, bool], list[str]]:
    """The jobs these changed paths need, and the paths no rule knows.

    The caller decides what an unknown path means (on a push to main: everything).
    """
    needed = nothing()
    unknown: list[str] = []
    for path in paths:
        path = path.strip()
        if not path:
            continue
        hit = False
        for job, pattern in INPUTS.items():
            if _match(pattern, path):
                needed[job] = True
                hit = True
        if _match(SCENE, path) or (_match(E2E_SPEC, path) and not _match(PAGE_SPECS, path)):
            needed["scene"] = True
        if not hit and not _match(INERT, path):
            unknown.append(path)
    needed["scene"] = needed["scene"] and needed["e2e"]
    return needed, unknown


def decide(
    event: str,
    changed: Sequence[str] | None,
    *,
    before: str = "",
    forced: bool = False,
    nightly_due: bool = True,
) -> tuple[dict[str, bool], str]:
    """The jobs to run for this event, and one line saying why.

    `changed` is None when the diff could not be computed.
    """
    if event == "schedule":
        if not nightly_due:
            return nothing(), "nightly: main has no commits since the last nightly"
        return everything(), "nightly: full run"
    if event == "pull_request":
        if changed is None:
            return everything(), "pull request: could not diff against the base; everything"
        needed, _ = classify(changed)
        # A pull request runs every job but the Playwright shards, as it always has, so an
        # unclassified path does not add them either.
        result = {**everything(), "e2e": needed["e2e"], "scene": needed["scene"]}
        return result, "pull request: every job, Playwright as the diff needs"
    if event == "push":
        if not before or before == ZERO_SHA:
            return everything(), "push: no `before` commit (new branch); everything"
        if forced:
            return everything(), "push: force-push; everything"
        if changed is None:
            return everything(), f"push: could not diff against {before}; everything"
        needed, unknown = classify(changed)
        if unknown:
            return everything(), f"push: unclassified paths {unknown[:5]}; everything"
        # Main is where the scene specs a page-only pull request skipped are run.
        needed["scene"] = needed["e2e"]
        return needed, "push: only the jobs whose inputs changed"
    return everything(), f"{event or 'unknown event'}: everything"


def outputs(needed: dict[str, bool]) -> str:
    lines = [f"{job}={'true' if needed[job] else 'false'}" for job in (*JOBS, "scene")]
    if not needed["e2e"]:
        # The shards job is skipped; a one-entry list keeps its matrix valid regardless.
        shards = [1]
    else:
        shards = [1, 2, 3, 4, 5, 6] if needed["scene"] else [1, 2, 3, 4]
    lines.append(f"shards={json.dumps(shards, separators=(',', ':'))}")
    return "\n".join(lines) + "\n"


def nightly_due(head: str, runs: Sequence[dict[str, object]]) -> bool:
    """Whether main has commits since the last nightly that ran.

    `runs` are this workflow's completed scheduled runs, newest first (the Actions API's
    `workflow_runs`). A cancelled or skipped one did not test anything, so it does not
    count; a failed one did, and is not repeated until main moves.
    """
    for run in runs:
        if run.get("conclusion") in ("cancelled", "skipped", "startup_failure", "stale"):
            continue
        return run.get("head_sha") != head
    return True


def _git_diff(before: str) -> list[str] | None:
    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=False)

    if git("fetch", "--no-tags", "--depth=1", "origin", before).returncode != 0:
        return None
    diff = git("diff", "--name-only", before, "HEAD")
    if diff.returncode != 0:
        return None
    return diff.stdout.splitlines()


def _last_nightlies() -> list[dict[str, object]] | None:
    try:
        url = (
            f"{os.environ.get('GITHUB_API_URL', 'https://api.github.com')}/repos/"
            f"{os.environ['GITHUB_REPOSITORY']}/actions/workflows/ci.yml/runs"
            "?event=schedule&status=completed&branch=main&per_page=20"
        )
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
                "Accept": "application/vnd.github+json",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return list(json.load(response)["workflow_runs"])
    except Exception as error:  # noqa: BLE001 - any failure here means "run"
        print(f"::warning::could not list earlier nightlies ({error!r}); running everything")
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--event", required=True)
    parser.add_argument("--before", default="", help="push: the commit the push replaced")
    parser.add_argument("--base", default="", help="pull_request: the base commit")
    parser.add_argument("--forced", default="false", help="push: 'true' for a force-push")
    parser.add_argument("--head", default="", help="schedule: the commit being tested")
    parser.add_argument(
        "--stdin", action="store_true", help="read changed paths from stdin instead of git"
    )
    args = parser.parse_args(argv)

    changed: list[str] | None = None
    due = True
    if args.stdin:
        changed = sys.stdin.read().splitlines()
    elif args.event == "pull_request" and args.base:
        changed = _git_diff(args.base)
    elif args.event == "push" and args.before not in ("", ZERO_SHA) and args.forced != "true":
        changed = _git_diff(args.before)
    elif args.event == "schedule":
        runs = _last_nightlies()
        due = True if runs is None else nightly_due(args.head, runs)

    if changed is not None:
        print("Changed:", *changed, sep="\n  ")
    needed, reason = decide(
        args.event, changed, before=args.before, forced=args.forced == "true", nightly_due=due
    )
    result = outputs(needed)
    print(reason)
    print(result, end="")

    if path := os.environ.get("GITHUB_OUTPUT"):
        with open(path, "a") as output:
            output.write(result)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(path, "a") as summary:
            summary.write(f"**{reason}**\n\n| job | runs |\n|---|---|\n")
            for job in (*JOBS, "scene"):
                summary.write(f"| {job} | {'yes' if needed[job] else 'no'} |\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
