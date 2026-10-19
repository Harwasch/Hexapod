"""Tests for ci_changes.py. Run: python3 -m unittest discover -s .github/scripts -v"""

from __future__ import annotations

import unittest

from ci_changes import JOBS, ZERO_SHA, decide, nightly_due, outputs

BEFORE = "1" * 40
NODE = {"web", "e2e"}


def push(*paths: str, before: str = BEFORE, forced: bool = False) -> set[str]:
    needed, _ = decide("push", list(paths), before=before, forced=forced)
    return {job for job in JOBS if needed[job]}


class PushToMain(unittest.TestCase):
    def test_docs_only_runs_nothing(self) -> None:
        self.assertEqual(push("docs/HANDOFF.md", "README.md", "docs/DECISIONS/0001.md"), set())

    def test_deployment_doc_runs_the_workflows_check_only(self) -> None:
        self.assertEqual(push("docs/DEPLOYMENT.md"), {"workflows"})

    def test_web_file_runs_web_and_all_six_playwright_shards(self) -> None:
        needed, _ = decide("push", ["apps/web/src/features/app/App.tsx"], before=BEFORE)
        self.assertEqual({job for job in JOBS if needed[job]}, NODE)
        self.assertTrue(needed["scene"])
        self.assertIn("shards=[1,2,3,4,5,6]", outputs(needed))

    def test_api_file_runs_api_and_image(self) -> None:
        self.assertEqual(push("apps/api/app/main.py"), {"api", "image"})

    def test_api_json_the_web_app_imports_runs_web_too(self) -> None:
        self.assertEqual(
            push("apps/api/app/seed/demo_ranch.json"), {"api", "image", "web", "e2e"}
        )

    def test_ci_yml_runs_the_workflows_check_without_playwright(self) -> None:
        self.assertEqual(push(".github/workflows/ci.yml"), {"workflows"})
        self.assertEqual(push(".github/scripts/ci_changes.py"), {"workflows"})

    def test_lockfile_runs_every_node_job(self) -> None:
        needed, _ = decide("push", ["pnpm-lock.yaml"], before=BEFORE)
        self.assertEqual({job for job in JOBS if needed[job]}, NODE)
        self.assertTrue(needed["scene"])

    def test_contracts_run_web_playwright_and_api(self) -> None:
        self.assertEqual(
            push("packages/contracts/openapi.json"), {"web", "e2e", "api"}
        )

    def test_root_lint_config_runs_web_not_playwright(self) -> None:
        self.assertEqual(push("eslint.config.js"), {"web"})
        self.assertEqual(push(".prettierrc"), {"web"})

    def test_python_projects(self) -> None:
        self.assertEqual(
            push("tools/captures/splat_tiles.py"), {"captures", "pipeline", "api", "image"}
        )
        self.assertEqual(
            push("tools/pipeline/stages.py"), {"pipeline", "trainer", "api", "image"}
        )
        self.assertEqual(push("workers/worlds/gateway.py"), {"worlds"})
        self.assertEqual(push("hardware/arm/armdesign/optimize.py"), {"arm"})
        self.assertEqual(push("fly.toml"), {"image"})

    def test_an_unclassified_path_runs_everything(self) -> None:
        self.assertEqual(push("docs/x.md", "brand-new-dir/thing.py"), set(JOBS))

    def test_all_zeros_before_runs_everything(self) -> None:
        needed, reason = decide("push", ["docs/x.md"], before=ZERO_SHA)
        self.assertTrue(all(needed.values()), reason)

    def test_force_push_runs_everything(self) -> None:
        self.assertEqual(push("docs/x.md", forced=True), set(JOBS))

    def test_a_diff_that_failed_runs_everything(self) -> None:
        needed, _ = decide("push", None, before=BEFORE)
        self.assertTrue(all(needed.values()))


class PullRequest(unittest.TestCase):
    def pr(self, *paths: str) -> dict[str, bool]:
        return decide("pull_request", list(paths))[0]

    def jobs(self, *paths: str) -> set[str]:
        needed = self.pr(*paths)
        return {job for job in JOBS if needed[job]}

    def test_docs_only_runs_nothing(self) -> None:
        self.assertEqual(self.jobs("docs/HANDOFF.md", "README.md"), set())

    def test_api_file_runs_api_and_image_only(self) -> None:
        self.assertEqual(self.jobs("apps/api/app/main.py"), {"api", "image"})

    def test_web_file_runs_web_and_playwright(self) -> None:
        self.assertEqual(self.jobs("apps/web/src/features/app/App.tsx"), NODE)

    def test_an_unclassified_path_runs_everything(self) -> None:
        self.assertEqual(self.jobs("brand-new-dir/thing.py"), set(JOBS))

    def test_a_diff_that_failed_runs_everything(self) -> None:
        self.assertTrue(all(decide("pull_request", None)[0].values()))

    def test_ci_yml_alone_no_longer_forces_playwright(self) -> None:
        self.assertFalse(self.pr(".github/workflows/ci.yml")["e2e"])

    def test_page_only_change_runs_four_shards(self) -> None:
        needed = self.pr("apps/web/src/features/app/App.tsx")
        self.assertTrue(needed["e2e"])
        self.assertFalse(needed["scene"])
        self.assertIn("shards=[1,2,3,4]", outputs(needed))

    def test_scene_change_runs_six_shards(self) -> None:
        self.assertTrue(self.pr("apps/web/src/cesium/globe.ts")["scene"])
        self.assertTrue(self.pr("apps/web/e2e/poke.spec.ts")["scene"])
        self.assertFalse(self.pr("apps/web/e2e/admin.spec.ts")["scene"])

    def test_backend_only_skips_playwright(self) -> None:
        needed = self.pr("apps/api/app/main.py", "tools/pipeline/stages.py")
        self.assertFalse(needed["e2e"])
        self.assertIn("shards=[1]", outputs(needed))


class Nightly(unittest.TestCase):
    def test_runs_everything_when_main_moved(self) -> None:
        self.assertTrue(nightly_due("b" * 40, [{"head_sha": "a" * 40, "conclusion": "success"}]))
        needed, _ = decide("schedule", None, nightly_due=True)
        self.assertTrue(all(needed.values()))

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
        needed, _ = decide("workflow_dispatch", None)
        self.assertTrue(all(needed.values()))


if __name__ == "__main__":
    unittest.main()
