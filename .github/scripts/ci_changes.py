"""Which CI jobs a change needs: the `changes` job of .github/workflows/ci.yml.

Each job in ci.yml is gated on one output of this script. The decision is made here
rather than inline in the workflow so it can be tested (test_ci_changes.py, run by the
workflows job) and run by hand:

    git diff --name-only origin/main HEAD | python3 .github/scripts/ci_changes.py --event push

By event:

- pull_request: only the jobs whose inputs changed between the base and the PR; the
  Playwright shards run the page specs alone when the 3D scene's inputs did not change.
  On a draft pull request, only the fast checks (`FAST`) of those; the rest run when it
  is marked ready for review, which starts a new run.
- push to main: only the jobs whose inputs changed between the push's `before` and its
  head, with the scene specs whenever Playwright runs. A new branch (an all-zeros
  `before`), a force-push, or a diff that cannot be computed runs everything.
- schedule (the nightly): everything, unless main's head is the commit the last nightly
  already tested -- then nothing.
- workflow_dispatch, or anything else: everything.

On a pull request or a push, a change to ci.yml or to this file runs everything too, so a
broken job step cannot merge unrun. Test: python3 -m unittest discover -s .github/scripts

The mapping is deliberately coarse and errs towards running: a path is matched against
every job's inputs (one file can need several jobs), a path no rule knows about runs
everything, and only `INERT` paths are allowed to run nothing. The nightly is the
backstop for a dependency the mapping misses.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import subprocess
import sys
import urllib.request
from collections.abc import Collection, Iterable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# The jobs, by output name. `e2e` gates the Playwright shards; `scene` says whether they
# include the 3D scene specs (six shards) or the app's page specs alone (four). `format`
# is `prettier --check .` alone, for a change Prettier reads that does not run the web job
# (which runs it too).
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
    "format",
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
    # load off disk.
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
    # packages/geo fixtures, packages/world, data/tiles, infra/, functions/r2 and two web
    # sources.
    "api": "|".join(
        [
            r"apps/api/",
            r"functions/",
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
            r"\.github/scripts/check_fly_toml\.py$",
            r"\.dockerignore$",
        ]
    ),
    # Captures · imports tools/pipeline's bridge; its tests read data/tiles, infra/modal
    # apps, the API's sidecars and legacy catalogue, and its fixture checksum must agree
    # with packages/world.
    "captures": "|".join(
        [
            r"tools/(captures|pipeline)/",
            r"data/tiles/",
            r"infra/modal/",
            r"packages/world/",
            r"apps/api/app/services/sidecars\.py$",
            r"apps/api/app/seed/legacy_captures\.json$",
        ]
    ),
    # Pipeline · imports tools/captures, builds infra/modal/app.py, renders data/tiles; its
    # tests read apps/api (test_experiments.py, test_cloud_stops.py).
    "pipeline": r"tools/(pipeline|captures)/|infra/modal/|data/tiles/|apps/api/",
    # Trainer · infra/modal's requirements and check, against tools/pipeline's argv.
    "trainer": r"tools/pipeline/|infra/modal/",
    # Worlds · the worker, scripts/ (its tests and the capabilities sync), and the web and
    # API files those read (prepare-worlds-land-release.py reads fly.toml and migrations).
    "worlds": "|".join(
        [
            r"workers/worlds/",
            r"scripts/",
            r"apps/web/src/worlds/",
            r"apps/web/package\.json$",
            r"apps/api/",
            r"fly\.toml$",
        ]
    ),
    "arm": r"hardware/arm/",
    # Workflows · parse every workflow, and check their secrets against the handover.
    "workflows": r"\.github/|docs/DEPLOYMENT\.md$",
}

# CI's own definition: a change to it runs every job, so a broken step cannot merge unrun.
CI_SELF = r"\.github/workflows/ci\.yml$|\.github/scripts/ci_changes\.py$"

# What `prettier --check .` reads, by extension (the root .prettierignore narrows it; this
# does not, which only means the format job runs a little more often than it must).
PRETTIER = (
    r".*\.(md|mdx|markdown|ya?ml|json|json5|jsonc|js|jsx|mjs|cjs|ts|tsx|mts|cts"
    r"|css|scss|less|html|vue|graphql|gql|hbs|handlebars)$|(.*/)?\.prettierrc"
)

# Paths no job reads: they may run nothing. Everything else that matches no rule above
# runs everything, so a new top-level directory or root file is never silently untested.
INERT = r"docs/|[^/]*\.md$|\.claude/|\.agents/|\.gitignore$|skills-lock\.json$|LICENSE$"

# The 3D scene specs (the `scene` and `webgpu` projects, apps/web/playwright.config.ts)
# render from these, and from every file the dev harnesses they load import (see
# `scene_sources`). A diff that reaches the e2e job but none of these runs the `chromium`
# project alone -- on a pull request. Main runs the scene specs whenever it runs
# Playwright at all.
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

# What a draft pull request runs: the checks that take a minute, not a queue. Twenty
# stacked drafts re-running every job on each push held main's own CI up for hours.
FAST = ("workflows", "format")
DRAFT_NOTE = "Draft PR: slow jobs skipped; mark ready to run the full suite"

# `import x from "y"`, `export * from "y"`, `import "y"`, `import("y")`, `new URL("y", ...)`.
_IMPORT = re.compile(r"""(?:\bfrom|\bimport|\bnew\s+URL)\s*\(?\s*["']([^"']+)["']""")
_SOURCE = (".ts", ".tsx", ".js", ".jsx", ".mjs")
_EXTENSIONS = ("", *_SOURCE, ".json", ".css")


def _resolve(base: Path) -> Path | None:
    for candidate in (
        *(base.with_name(base.name + ext) for ext in _EXTENSIONS),
        *(base / f"index{ext}" for ext in _SOURCE),
    ):
        if candidate.is_file():
            return candidate
    return None


@functools.cache
def scene_sources(root: Path = ROOT) -> tuple[set[str], set[str]]:
    """What the scene specs' pages are built from: (files, package directories).

    Every scene spec drives a dev harness, apps/web/src/dev/*Harness.ts, so this follows
    their imports through apps/web/src (relative and `@/` imports, transitively) and notes
    each `@twin/<name>` package reached. Bare npm imports are covered by the lockfile.
    """
    root = root.resolve()
    src = root / "apps/web/src"
    todo = sorted(path for path in (src / "dev").glob("*") if path.suffix in _SOURCE)
    files: set[str] = set()
    packages: set[str] = set()
    while todo:
        path = todo.pop().resolve()
        relative = path.relative_to(root).as_posix()
        if relative in files:
            continue
        files.add(relative)
        if path.suffix not in _SOURCE:
            continue
        for spec in _IMPORT.findall(path.read_text(encoding="utf-8", errors="replace")):
            spec = spec.split("?", 1)[0]
            if spec.startswith("@twin/"):
                packages.add(f"packages/{spec.split('/')[1]}/")
                continue
            if spec.startswith("@/"):
                base = src / spec[2:]
            elif spec.startswith("."):
                base = path.parent / spec
            else:
                continue
            target = _resolve(base)
            if target is not None and root in target.resolve().parents:
                todo.append(target)
    return files, packages


def _match(pattern: str, path: str) -> bool:
    return re.match(f"(?:{pattern})", path) is not None


def everything() -> dict[str, bool]:
    # The web job runs `prettier --check .` itself, so the format job is not needed.
    return {**{job: True for job in JOBS}, "format": False, "scene": True}


def nothing() -> dict[str, bool]:
    return {**{job: False for job in JOBS}, "scene": False}


def classify(
    paths: Iterable[str], scene: tuple[Collection[str], Collection[str]] | None = None
) -> tuple[dict[str, bool], list[str]]:
    """The jobs these changed paths need, and the paths no rule knows.

    The caller decides what an unknown path means (everything). `scene` is
    `scene_sources()`'s answer, computed from the checkout when not given.
    """
    scene_files, scene_packages = scene_sources() if scene is None else scene
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
        if (
            _match(SCENE, path)
            or path in scene_files
            or path.startswith(tuple(scene_packages))
            or (_match(E2E_SPEC, path) and not _match(PAGE_SPECS, path))
        ):
            needed["scene"] = True
        if _match(PRETTIER, path):
            needed["format"] = True
        if not hit and not _match(INERT, path):
            unknown.append(path)
    needed["scene"] = needed["scene"] and needed["e2e"]
    needed["format"] = needed["format"] and not needed["web"]
    return needed, unknown


def draft_only(needed: dict[str, bool]) -> dict[str, bool]:
    """A draft's jobs: the fast ones of `needed` alone. The web job checks formatting
    itself, so where it would have run, the format job runs instead."""
    fast = {**nothing(), **{job: needed[job] for job in FAST}}
    fast["format"] = needed["format"] or needed["web"]
    return fast


def _overrides(paths: Sequence[str], unknown: Sequence[str]) -> str | None:
    """Why this diff must run everything, if it must."""
    own = [path.strip() for path in paths if _match(CI_SELF, path.strip())]
    if own:
        return f"CI's own definition changed ({', '.join(own)})"
    if unknown:
        return f"unclassified paths {list(unknown[:5])}"
    return None


def decide(
    event: str,
    changed: Sequence[str] | None,
    *,
    before: str = "",
    forced: bool = False,
    nightly_due: bool = True,
    draft: bool = False,
    scene: tuple[Collection[str], Collection[str]] | None = None,
) -> tuple[dict[str, bool], str]:
    """The jobs to run for this event, and one line saying why.

    `changed` is None when the diff could not be computed. `draft` counts on a pull
    request alone: a push, a dispatch and the nightly are never drafts.
    """
    if draft and event == "pull_request":
        needed, why = decide(event, changed, scene=scene)
        return draft_only(needed), f"draft {why}; the fast checks alone"
    if event == "schedule":
        if not nightly_due:
            return nothing(), "nightly: main has no commits since the last nightly"
        return everything(), "nightly: full run"
    if event == "pull_request":
        if changed is None:
            return everything(), "pull request: could not diff against the base; everything"
        needed, unknown = classify(changed, scene)
        if why := _overrides(changed, unknown):
            return everything(), f"pull request: {why}; everything"
        # Unlike a push, a page-only pull request runs the app's page specs alone.
        return needed, "pull request: only the jobs whose inputs changed"
    if event == "push":
        if not before or before == ZERO_SHA:
            return everything(), "push: no `before` commit (new branch); everything"
        if forced:
            return everything(), "push: force-push; everything"
        if changed is None:
            return everything(), f"push: could not diff against {before}; everything"
        needed, unknown = classify(changed, scene)
        if why := _overrides(changed, unknown):
            return everything(), f"push: {why}; everything"
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


def _git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False, cwd=cwd)


def diff_names(before: str, head: str = "HEAD", cwd: Path | None = None) -> list[str] | None:
    """The paths that differ between two commits. `--no-renames`, so a rename lists its
    source as well as its destination: a job that read the old path must run too."""
    diff = _git("diff", "--no-renames", "--name-only", before, head, cwd=cwd)
    if diff.returncode != 0:
        return None
    return diff.stdout.splitlines()


def _git_diff(before: str) -> list[str] | None:
    if _git("fetch", "--no-tags", "--depth=1", "origin", before).returncode != 0:
        return None
    return diff_names(before)


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
    parser.add_argument("--draft", default="false", help="pull_request: 'true' for a draft")
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
    draft = args.event == "pull_request" and args.draft == "true"
    needed, reason = decide(
        args.event,
        changed,
        before=args.before,
        forced=args.forced == "true",
        nightly_due=due,
        draft=draft,
    )
    result = outputs(needed)
    print(reason)
    print(result, end="")

    if path := os.environ.get("GITHUB_OUTPUT"):
        with open(path, "a") as output:
            output.write(result)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(path, "a") as summary:
            if draft:
                summary.write(f"{DRAFT_NOTE}.\n\n")
            summary.write(f"**{reason}**\n\n| job | runs |\n|---|---|\n")
            for job in (*JOBS, "scene"):
                summary.write(f"| {job} | {'yes' if needed[job] else 'no'} |\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
