"""Tests for ci_changes.py. Run: python3 -m unittest discover -s .github/scripts"""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import yaml

from ci_changes import (
    DRAFT_NOTE,
    FAST,
    JOBS,
    ROOT,
    ZERO_SHA,
    decide,
    diff_names,
    main,
    nightly_due,
    outputs,
)

BEFORE = "1" * 40
NODE = {"web", "e2e"}
# Everything: every job but `format`, whose `prettier --check .` the web job runs itself.
FULL = set(JOBS) - {"format"}
# A web page no dev harness imports: on a pull request, the page specs alone.
PAGE_ONLY = "apps/web/src/admin/AdminApp.tsx"


def ran(needed: dict[str, bool]) -> set[str]:
    return {job for job in JOBS if needed[job]}


def push(*paths: str, before: str = BEFORE, forced: bool = False) -> set[str]:
    return ran(decide("push", list(paths), before=before, forced=forced)[0])


def pr(*paths: str) -> dict[str, bool]:
    return decide("pull_request", list(paths))[0]


class PushToMain(unittest.TestCase):
    def test_docs_only_runs_the_format_check_alone(self) -> None:
        self.assertEqual(
            push("docs/HANDOFF.md", "README.md", "docs/DECISIONS/0001.md"), {"format"}
        )

    def test_deployment_doc_runs_the_workflows_and_format_checks(self) -> None:
        self.assertEqual(push("docs/DEPLOYMENT.md"), {"workflows", "format"})

    def test_web_file_runs_web_and_all_six_playwright_shards(self) -> None:
        needed, _ = decide("push", [PAGE_ONLY], before=BEFORE)
        self.assertEqual(ran(needed), NODE)
        self.assertTrue(needed["scene"])
        self.assertIn("shards=[1,2,3,4,5,6]", outputs(needed))

    def test_api_file_runs_api_image_pipeline_and_worlds(self) -> None:
        self.assertEqual(
            push("apps/api/app/main.py"), {"api", "image", "pipeline", "worlds"}
        )

    def test_api_json_the_web_app_imports_runs_web_too(self) -> None:
        self.assertTrue(
            NODE <= push("apps/api/app/seed/demo_ranch.json"),
        )

    def test_lockfile_runs_every_node_job(self) -> None:
        needed, _ = decide("push", ["pnpm-lock.yaml"], before=BEFORE)
        self.assertEqual(ran(needed), NODE)
        self.assertTrue(needed["scene"])

    def test_contracts_run_web_playwright_and_api(self) -> None:
        self.assertEqual(push("packages/contracts/openapi.json"), {"web", "e2e", "api"})

    def test_root_lint_config_runs_web_not_playwright(self) -> None:
        self.assertEqual(push("eslint.config.js"), {"web"})
        self.assertEqual(push(".prettierrc"), {"web"})

    def test_python_projects(self) -> None:
        self.assertEqual(
            push("tools/captures/splat_tiles.py"), {"captures", "pipeline", "api", "image"}
        )
        self.assertEqual(
            push("tools/pipeline/stages.py"),
            {"captures", "pipeline", "trainer", "api", "image"},
        )
        self.assertEqual(push("workers/worlds/gateway.py"), {"worlds"})
        self.assertEqual(push("hardware/arm/armdesign/optimize.py"), {"arm"})

    def test_an_unclassified_path_runs_everything(self) -> None:
        self.assertEqual(push("docs/x.md", "brand-new-dir/thing.py"), FULL)

    def test_all_zeros_before_runs_everything(self) -> None:
        needed, reason = decide("push", ["docs/x.md"], before=ZERO_SHA)
        self.assertEqual(ran(needed), FULL, reason)

    def test_force_push_runs_everything(self) -> None:
        self.assertEqual(push("docs/x.md", forced=True), FULL)

    def test_a_diff_that_failed_runs_everything(self) -> None:
        self.assertEqual(ran(decide("push", None, before=BEFORE)[0]), FULL)


class CrossProjectReads(unittest.TestCase):
    """Files one project's tests read out of another (review of PR #44)."""

    def test_functions_run_the_api(self) -> None:
        # apps/api/tests/test_worker_outputs.py and test_sidecar_attach.py read it.
        self.assertIn("api", push("functions/r2/[[path]].js"))

    def test_api_runs_the_pipeline(self) -> None:
        # tools/pipeline/tests/test_experiments.py reads apps/api/app/api/v1/phone.py.
        self.assertIn("pipeline", push("apps/api/app/api/v1/phone.py"))

    def test_legacy_captures_runs_captures(self) -> None:
        # tools/captures/tests/test_real_tree.py reads it.
        self.assertIn("captures", push("apps/api/app/seed/legacy_captures.json"))

    def test_fly_toml_runs_worlds(self) -> None:
        # scripts/prepare-worlds-land-release.py reads it; scripts/tests test that.
        self.assertEqual(push("fly.toml"), {"image", "worlds"})

    def test_pipeline_runs_captures(self) -> None:
        # tools/captures imports tools/pipeline/captures_bridge.py and cloud.py.
        self.assertIn("captures", push("tools/pipeline/captures_bridge.py"))


class Renames(unittest.TestCase):
    def test_the_diff_lists_a_rename_source_too(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)

            def git(*args: str) -> None:
                subprocess.run(
                    ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                    cwd=repo,
                    check=True,
                    capture_output=True,
                )

            git("init", "-q")
            (repo / "data/tiles").mkdir(parents=True)
            (repo / "data/tiles/captures.json").write_text('{"captures": [1, 2, 3]}\n')
            git("add", "-A")
            git("commit", "-qm", "one")
            (repo / "apps/api/app/seed").mkdir(parents=True)
            git("mv", "data/tiles/captures.json", "apps/api/app/seed/legacy_captures.json")
            git("commit", "-qm", "two")
            changed = diff_names("HEAD~1", "HEAD", cwd=repo)
        self.assertEqual(
            sorted(changed or []),
            ["apps/api/app/seed/legacy_captures.json", "data/tiles/captures.json"],
        )

    def test_commit_6fbad71b_runs_captures_and_pipeline(self) -> None:
        # data/tiles/captures.json became apps/api/app/seed/legacy_captures.json there.
        jobs = push("data/tiles/captures.json", "apps/api/app/seed/legacy_captures.json")
        self.assertTrue({"captures", "pipeline"} <= jobs, jobs)


class CiItself(unittest.TestCase):
    def test_ci_yml_runs_every_job_playwright_included(self) -> None:
        for path in (".github/workflows/ci.yml", ".github/scripts/ci_changes.py"):
            needed, _ = decide("push", [path], before=BEFORE)
            self.assertEqual(ran(needed), FULL, path)
            self.assertTrue(needed["scene"], path)
            self.assertTrue(pr(path)["e2e"], path)

    def test_other_workflows_run_the_workflows_check(self) -> None:
        self.assertEqual(push(".github/workflows/deploy.yml"), {"workflows", "format"})
        self.assertEqual(push(".github/scripts/test_ci_changes.py"), {"workflows"})


class Format(unittest.TestCase):
    def test_a_file_prettier_reads_outside_the_web_job_runs_the_format_job(self) -> None:
        self.assertEqual(push("hardware/arm/data/tasks.yaml"), {"arm", "format"})
        self.assertEqual(push("infra/cors/upload.json"), {"api", "image", "format"})
        self.assertIn("format", ran(pr("workers/worlds/README.md")))

    def test_python_alone_does_not(self) -> None:
        self.assertNotIn("format", push("tools/pipeline/stages.py"))

    def test_the_web_job_makes_it_redundant(self) -> None:
        self.assertNotIn("format", push("docs/x.md", PAGE_ONLY))


class PullRequest(unittest.TestCase):
    def test_docs_only_runs_nothing_heavy(self) -> None:
        self.assertEqual(ran(pr("docs/HANDOFF.md", "README.md")), {"format"})

    def test_api_file_runs_api_and_image_and_their_readers(self) -> None:
        self.assertEqual(
            ran(pr("apps/api/app/main.py")), {"api", "image", "pipeline", "worlds"}
        )

    def test_web_file_runs_web_and_playwright(self) -> None:
        self.assertEqual(ran(pr(PAGE_ONLY)), NODE)

    def test_an_unclassified_path_runs_everything(self) -> None:
        self.assertEqual(ran(pr("brand-new-dir/thing.py")), FULL)

    def test_a_diff_that_failed_runs_everything(self) -> None:
        self.assertEqual(ran(decide("pull_request", None)[0]), FULL)

    def test_page_only_change_runs_four_shards(self) -> None:
        needed = pr(PAGE_ONLY)
        self.assertTrue(needed["e2e"])
        self.assertFalse(needed["scene"])
        self.assertIn("shards=[1,2,3,4]", outputs(needed))

    def test_scene_change_runs_six_shards(self) -> None:
        self.assertTrue(pr("apps/web/src/cesium/globe.ts")["scene"])
        self.assertTrue(pr("apps/web/e2e/poke.spec.ts")["scene"])
        self.assertFalse(pr("apps/web/e2e/admin.spec.ts")["scene"])

    def test_what_a_harness_imports_is_a_scene_input(self) -> None:
        # apps/web/src/dev/sceneSelectHarness.ts imports both, outside the scene's dirs.
        self.assertTrue(pr("apps/web/src/features/sites/ObjectCard.tsx")["scene"])
        self.assertTrue(pr("apps/web/src/styles/mission.css")["scene"])
        self.assertTrue(pr("packages/ui/src/index.ts")["scene"])

    def test_backend_only_skips_playwright(self) -> None:
        needed = pr("apps/api/app/main.py", "tools/pipeline/stages.py")
        self.assertFalse(needed["e2e"])
        self.assertIn("shards=[1]", outputs(needed))


class Nightly(unittest.TestCase):
    def test_runs_everything_when_main_moved(self) -> None:
        self.assertTrue(nightly_due("b" * 40, [{"head_sha": "a" * 40, "conclusion": "success"}]))
        self.assertEqual(ran(decide("schedule", None, nightly_due=True)[0]), FULL)

    def test_runs_nothing_when_main_did_not_move(self) -> None:
        runs = [{"head_sha": "a" * 40, "conclusion": "failure"}]
        self.assertFalse(nightly_due("a" * 40, runs))
        needed, _ = decide("schedule", None, nightly_due=False)
        self.assertFalse(any(needed.values()))

    def test_a_cancelled_nightly_does_not_count(self) -> None:
        runs = [
            {"head_sha": "b" * 40, "conclusion": "cancelled"},
            {"head_sha": "a" * 40, "conclusion": "success"},
        ]
        self.assertTrue(nightly_due("b" * 40, runs))

    def test_the_first_nightly_runs(self) -> None:
        self.assertTrue(nightly_due("a" * 40, []))


class Dispatch(unittest.TestCase):
    def test_dispatch_runs_everything(self) -> None:
        self.assertEqual(ran(decide("workflow_dispatch", None)[0]), FULL)


class Draft(unittest.TestCase):
    """A draft pull request runs the fast checks alone; marking it ready runs the rest."""

    def draft(self, *paths: str) -> dict[str, bool]:
        return decide("pull_request", list(paths), draft=True)[0]

    def test_a_draft_runs_only_fast_jobs(self) -> None:
        for paths in (
            [".github/workflows/ci.yml"],
            ["brand-new-dir/thing.py"],
            ["apps/api/app/main.py", "apps/web/src/cesium/globe.ts"],
            [PAGE_ONLY],
            ["hardware/arm/data/tasks.yaml"],
        ):
            with self.subTest(paths=paths):
                needed = self.draft(*paths)
                self.assertLessEqual(ran(needed), set(FAST))
                self.assertFalse(needed["scene"])
                self.assertIn("shards=[1]", outputs(needed))

    def test_a_draft_still_runs_the_fast_jobs_its_diff_needs(self) -> None:
        self.assertEqual(ran(self.draft(".github/workflows/ci.yml")), {"workflows", "format"})
        self.assertEqual(ran(self.draft("docs/DEPLOYMENT.md")), {"workflows", "format"})
        self.assertEqual(ran(self.draft("tools/pipeline/stages.py")), set())

    def test_a_draft_checks_formatting_where_the_web_job_would_have(self) -> None:
        self.assertEqual(ran(self.draft(PAGE_ONLY)), {"format"})

    def test_a_diff_that_failed_runs_the_fast_jobs_on_a_draft(self) -> None:
        self.assertEqual(ran(decide("pull_request", None, draft=True)[0]), set(FAST))

    def test_a_ready_pull_request_is_unchanged(self) -> None:
        for paths in (
            [".github/workflows/ci.yml"],
            ["apps/api/app/main.py"],
            [PAGE_ONLY],
            ["docs/HANDOFF.md"],
        ):
            with self.subTest(paths=paths):
                self.assertEqual(
                    decide("pull_request", paths, draft=False), decide("pull_request", paths)
                )

    def test_push_dispatch_and_nightly_are_never_drafts(self) -> None:
        self.assertEqual(
            decide("push", ["apps/api/app/main.py"], before=BEFORE, draft=True),
            decide("push", ["apps/api/app/main.py"], before=BEFORE),
        )
        self.assertEqual(ran(decide("workflow_dispatch", None, draft=True)[0]), FULL)
        self.assertEqual(ran(decide("schedule", None, draft=True)[0]), FULL)

    def test_the_command_line_ignores_draft_outside_a_pull_request(self) -> None:
        for event, extra, expected in (
            ("pull_request", ["--draft", "true"], "api=false"),
            ("pull_request", ["--draft", "false"], "api=true"),
            ("push", ["--draft", "true", "--before", BEFORE], "api=true"),
            ("workflow_dispatch", ["--draft", "true"], "api=true"),
        ):
            with self.subTest(event=event, extra=extra):
                out = io.StringIO()
                with (
                    mock.patch.dict(os.environ, clear=False) as env,
                    mock.patch("sys.stdin", io.StringIO("apps/api/app/main.py\n")),
                    redirect_stdout(out),
                ):
                    env.pop("GITHUB_OUTPUT", None)
                    env.pop("GITHUB_STEP_SUMMARY", None)
                    main(["--event", event, "--stdin", *extra])
                self.assertIn(expected, out.getvalue())

    def test_the_summary_says_why_on_a_draft(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "summary.md"
            with (
                mock.patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(summary)}) as env,
                mock.patch("sys.stdin", io.StringIO("apps/api/app/main.py\n")),
                redirect_stdout(io.StringIO()),
            ):
                env.pop("GITHUB_OUTPUT", None)
                main(["--event", "pull_request", "--stdin", "--draft", "true"])
            self.assertTrue(summary.read_text().startswith(DRAFT_NOTE))


class Workflow(unittest.TestCase):
    """ci.yml's gates: every job runs when the changes job fails (fail open)."""

    def setUp(self) -> None:
        text = (ROOT / ".github/workflows/ci.yml").read_text()
        self.jobs = yaml.safe_load(text)["jobs"]

    def test_every_output_is_wired_through(self) -> None:
        outputs_ = self.jobs["changes"]["outputs"]
        self.assertEqual(set(outputs_), {*JOBS, "scene", "shards"})

    def test_every_gated_job_runs_when_changes_failed(self) -> None:
        for name, job in self.jobs.items():
            if name in ("changes", "e2e"):
                continue
            with self.subTest(job=name):
                self.assertIn("changes", job["needs"])
                self.assertIn("!cancelled()", job["if"])
                self.assertIn("!= 'false'", job["if"])
                self.assertNotIn("== 'true'", job["if"])

    def test_a_pull_request_reruns_when_marked_ready_and_says_whether_it_is_a_draft(
        self,
    ) -> None:
        text = (ROOT / ".github/workflows/ci.yml").read_text()
        # PyYAML reads the `on:` key as True.
        triggers = yaml.safe_load(text)[True]
        self.assertEqual(
            set(triggers["pull_request"]["types"]),
            {"opened", "synchronize", "reopened", "ready_for_review"},
        )
        (step,) = [s for s in self.jobs["changes"]["steps"] if s.get("id") == "filter"]
        self.assertEqual(step["env"]["DRAFT"], "${{ github.event.pull_request.draft }}")
        self.assertIn('--draft "$DRAFT"', step["run"])

    def test_the_shard_matrix_has_a_fallback(self) -> None:
        shard = self.jobs["e2e-shard"]
        self.assertIn("||", shard["strategy"]["matrix"]["shard"])
        (listing,) = [s for s in shard["steps"] if s.get("name") == "This shard's tests"]
        self.assertIn('"$SCENE" = "false"', listing["run"])


if __name__ == "__main__":
    unittest.main()
