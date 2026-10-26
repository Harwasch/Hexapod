"""Run the workflow's actual bash steps with fake AWS/GPG, never production credentials."""

import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

import yaml

from ci_changes import decide

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/export-run-inputs.yml"
JOB = "ec644db5-ceaf-4c01-bf89-b261623327c2"
PREFIX = f"runs/{JOB}/"
GENERATION = PREFIX + "p0123456789abcdef/"
ITEMS = "normalize/frames/ normalize/frames_meta.json pose/poses/ place/placement.json"
FINGERPRINTS = (
    "25398747159F3E911B4DAD38FEC5E5D75D14CBAE\n"
    "911483FD04C64E81117807D371D3ED3B448CE7B7"
)
PAYLOAD = "PRIVATE PHOTO CONTENT"

AWS = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open("aws-calls", "a") as log:
    log.write(json.dumps(args) + "\n")
operation = args[args.index("s3api") + 1]
assert operation in ("list-objects-v2", "get-object"), args
assert args[args.index("--bucket") + 1] == "private-test-bucket"
assert args[args.index("--region") + 1] == "auto"
assert "--no-paginate" not in args and "--max-items" not in args
objects = json.loads(Path("objects.json").read_text())
if os.environ.get("FAIL_AWS") == operation:
    print("PRIVATE RESPONSE BODY")
    print("PRIVATE RESPONSE BODY", file=sys.stderr)
    sys.exit(2)
if operation == "list-objects-v2":
    prefix = args[args.index("--prefix") + 1]
    keys = sorted(key for key in objects if key.startswith(prefix))
    if "--delimiter" in args:
        prefixes = sorted({prefix + key[len(prefix):].split("/")[0] + "/"
                           for key in keys if "/" in key[len(prefix):]})
        print(json.dumps({"CommonPrefixes": [{"Prefix": p} for p in prefixes]}))
    else:
        print(json.dumps({"Contents": [
            {"Key": key, "Size": len(objects[key].encode()), "ETag": '"test-etag"'}
            for key in keys]}))
else:
    key = args[args.index("--key") + 1]
    assert args[args.index("--if-match") + 1] == '"test-etag"'
    Path(args[-1]).write_text(objects[key])
    print(json.dumps({"Metadata": {"private": "PRIVATE RESPONSE BODY"}}))
'''

GPG = r'''#!/usr/bin/env python3
import json, os, shutil, sys
from pathlib import Path
args = sys.argv[1:]
with open("gpg-calls", "a") as log:
    log.write(json.dumps(args) + "\n")
if "--show-keys" in args:
    for fingerprint in os.environ["FAKE_FINGERPRINTS"].splitlines():
        print("fpr:::::::::" + fingerprint + ":")
else:
    assert "--encrypt" in args and "--batch" in args
    assert args[args.index("--trust-model") + 1] == "always"
    assert args[args.index("--recipient-file") + 1] == ".github/keys/run-export.asc"
    target = Path(args[args.index("--output") + 1])
    if os.environ.get("FAIL_GPG"):
        target.write_text("incomplete ciphertext")
        print("PRIVATE GPG ERROR", file=sys.stderr)
        sys.exit(2)
    # Keep the tar readable in fake-only tests to inspect exactly what was packed.
    shutil.copyfile(args[-1], target)
'''


class ExportRunInputsTests(unittest.TestCase):
    def setUp(self):
        self.workflow = yaml.safe_load(WORKFLOW.read_text())
        self.job = self.workflow["jobs"]["export"]
        self.steps = {step.get("id"): step for step in self.job["steps"]}
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp = Path(temporary.name)
        for directory in ("bin", ".github/scripts", ".github/keys", "home"):
            (self.tmp / directory).mkdir(parents=True)
        for name in ("scripts/export_run_inputs.py", "keys/run-export.asc"):
            shutil.copyfile(ROOT / ".github" / name, self.tmp / ".github" / name)
        for name, source in (("aws", AWS), ("gpg", GPG)):
            target = self.tmp / "bin" / name
            target.write_text(source)
            target.chmod(0o700)
        self.env = {
            "PATH": f"{self.tmp / 'bin'}:{os.environ['PATH']}",
            "HOME": str(self.tmp / "home"),
            "GITHUB_WORKSPACE": str(self.tmp),
            "GITHUB_OUTPUT": str(self.tmp / "outputs"),
            "RUN_PREFIX": PREFIX,
            "EXPORT_ITEMS": ITEMS,
            "OBJECT_STORAGE_ENDPOINT_URL": "https://storage.invalid",
            "OBJECT_STORAGE_BUCKET": "private-test-bucket",
            "FAKE_FINGERPRINTS": FINGERPRINTS,
        }
        self.objects()

    def objects(self, prefix=PREFIX, extra=None):
        objects = {
            prefix + "normalize/frames/private-photo.jpg": PAYLOAD,
            prefix + "normalize/frames/nested/another.jpg": PAYLOAD,
            prefix + "normalize/frames_meta.json": PAYLOAD,
            prefix + "pose/poses/images.bin": PAYLOAD,
            prefix + "place/placement.json": PAYLOAD,
        }
        objects.update(extra or {})
        (self.tmp / "objects.json").write_text(json.dumps(objects))
        return objects

    def run_step(self, step_id, **env):
        (self.tmp / "outputs").write_text("")
        result = subprocess.run(
            ["bash", "-e", "-c", self.steps[step_id]["run"]],
            cwd=self.tmp, env={**self.env, **env}, capture_output=True, text=True,
        )
        outputs = dict(line.split("=", 1) for line in (self.tmp / "outputs").read_text().splitlines())
        return result, outputs

    def plan(self, run=JOB, items=ITEMS):
        return self.run_step("plan", RUN=run, ITEMS=items)

    def calls(self, tool):
        path = self.tmp / f"{tool}-calls"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def assert_clean(self, result, success=False):
        self.assertFalse((self.tmp / ".run-export-private").exists())
        if not success:
            self.assertFalse((self.tmp / ".run-export-artifact").exists())
        for private in (PAYLOAD, "private-photo.jpg", "PRIVATE RESPONSE BODY", "PRIVATE GPG ERROR"):
            self.assertNotIn(private, result.stdout + result.stderr)

    def test_dispatch_only_minimal_permissions_and_ci_mapping(self):
        self.assertEqual(set(self.workflow.get("on", self.workflow.get(True))), {"workflow_dispatch"})
        self.assertEqual(self.workflow["permissions"], {"contents": "read"})
        self.assertNotIn("permissions", self.job)
        self.assertEqual(self.job["runs-on"], "ubuntu-latest")
        self.assertGreater(self.job["timeout-minutes"], 0)
        self.assertFalse(self.workflow["concurrency"]["cancel-in-progress"])
        self.assertEqual(self.job["steps"][0]["id"], "plan")
        for step in self.job["steps"]:
            if "run" in step:
                self.assertNotIn("${{", step["run"])
                self.assertNotIn("GITHUB_STEP_SUMMARY", step["run"])
        for path in (
            ".github/workflows/export-run-inputs.yml", ".github/keys/run-export.asc",
            ".github/scripts/export_run_inputs.py", ".github/scripts/test_export_run_inputs.py",
        ):
            needed, _ = decide("pull_request", [path])
            self.assertTrue(needed["workflows"], path)

    def test_valid_runs_and_subsets(self):
        for run, prefix in ((JOB, PREFIX), (PREFIX, PREFIX), (PREFIX[:-1], PREFIX),
                            (GENERATION, GENERATION), (GENERATION[:-1], GENERATION)):
            result, output = self.plan(run=run)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(output, {"prefix": prefix, "items": ITEMS})
        result, output = self.plan(items="pose/poses/ place/placement.json")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output["items"], "pose/poses/ place/placement.json")

    def test_hostile_runs_and_items_produce_no_outputs_or_commands(self):
        bad_runs = (
            "", " ", "*", JOB.upper(), "-" * 36, JOB + "/", PREFIX + "../",
            PREFIX + "p123/", GENERATION + "extra/", PREFIX + "\nitems=place/placement.json",
            JOB + ";touch PWNED", "$(touch PWNED)", "`touch PWNED`", "/" + PREFIX,
            PREFIX + "*", JOB + "\n", "--help", JOB + "'", PREFIX + "/",
        )
        bad_items = (
            "", " ", "*", "normalize/frames/*", "normalize/frames", "/pose/poses/",
            "../place/placement.json", "normalize/frames/../../secret", "--recursive",
            "pose/poses/;touch PWNED", "$(touch PWNED)", "`touch PWNED`",
            "pose/poses/\nother=value", "pose/poses/\tplace/placement.json",
            "pose/poses/ pose/poses/", "place/placement.json.bak", "pose/poses/\\",
        )
        for field, values in (("run", bad_runs), ("items", bad_items)):
            for value in values:
                with self.subTest(field=field, value=value):
                    result, output = self.plan(**{field: value})
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(output, {})
                    self.assertFalse((self.tmp / "PWNED").exists())
                    self.assertEqual(self.calls("aws"), [])
                    self.assertEqual(self.calls("gpg"), [])

    def test_missing_secrets_are_named_without_printing_values(self):
        names = ("CLOUDFLARE_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")
        for missing in names:
            env = {name: "SECRET-VALUE" for name in names if name != missing}
            result, _ = self.run_step("secrets", **env)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(missing, result.stdout)
            self.assertNotIn("SECRET-VALUE", result.stdout + result.stderr)
        result, _ = self.run_step("secrets", **dict.fromkeys(names, "SECRET-VALUE"))
        self.assertEqual(result.returncode, 0)

    def test_swapped_or_appended_key_refused_before_download(self):
        for fingerprints in ("0" * 40, FINGERPRINTS + "\n" + "0" * 40,
                             FINGERPRINTS.splitlines()[0] + "\n" + "0" * 40):
            with self.subTest(fingerprints=fingerprints):
                result, _ = self.run_step("encrypt", FAKE_FINGERPRINTS=fingerprints)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("fingerprint mismatch", result.stdout)
                self.assertEqual(self.calls("aws"), [])
                self.assert_clean(result)

    def test_root_export_packs_only_allowed_objects_and_cleans_before_upload(self):
        self.objects(extra={PREFIX + "train/private.ply": PAYLOAD,
                            PREFIX + "normalize/frames_meta.json.bak": PAYLOAD,
                            PREFIX + "normalize/frames/": ""})
        result, _ = self.run_step("encrypt")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"Selected prefix: {PREFIX}", result.stdout)
        self.assertIn("Downloaded 5 objects", result.stdout)
        self.assert_clean(result, success=True)
        with tarfile.open(self.tmp / ".run-export-artifact/run-inputs.tar.gpg") as archive:
            members = [member for member in archive.getmembers() if member.isfile()]
            self.assertEqual(len(members), 5)
            self.assertTrue(all(archive.extractfile(member).read() == PAYLOAD.encode() for member in members))
            self.assertFalse(any("train" in member.name or ".bak" in member.name for member in members))
        self.assertTrue(all("s3api" in args for args in self.calls("aws")))

    def test_auto_discovers_a_single_generation(self):
        self.objects(prefix=GENERATION)
        result, _ = self.run_step("encrypt")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(f"Selected prefix: {GENERATION}", result.stdout)
        self.assert_clean(result, success=True)

    def test_explicit_generation_ignores_other_generations_and_root(self):
        root_objects = self.objects()
        self.objects(prefix=GENERATION, extra=root_objects)
        result, _ = self.run_step("encrypt", RUN_PREFIX=GENERATION, EXPORT_ITEMS="pose/poses/")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_clean(result, success=True)
        for args in self.calls("aws"):
            option = "--key" if "get-object" in args else "--prefix"
            self.assertTrue(args[args.index(option) + 1].startswith(GENERATION))

    def test_ambiguous_root_or_generations_fail_before_get(self):
        for first in (PREFIX, PREFIX + "pfedcba9876543210/"):
            other = self.objects(prefix=first)
            self.objects(prefix=GENERATION, extra=other)
            result, _ = self.run_step("encrypt")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Multiple populated prefixes", result.stderr)
            self.assertFalse(any("get-object" in args for args in self.calls("aws")))
            self.assert_clean(result)

    def test_missing_items_and_empty_runs_fail(self):
        for objects, message in (({}, "No requested inputs"),
                                 ({PREFIX + "place/placement.json": PAYLOAD}, "requested item is missing")):
            (self.tmp / "objects.json").write_text(json.dumps(objects))
            result, _ = self.run_step("encrypt")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(message, result.stderr)
            self.assertFalse(any("get-object" in args for args in self.calls("aws")))
            self.assert_clean(result)

    def test_unsafe_storage_paths_fail_before_get(self):
        for suffix in ("../../escape", "nested/../../../escape", "a\\b", "a\nb", "./a", "a//b"):
            self.objects(extra={PREFIX + "normalize/frames/" + suffix: PAYLOAD})
            result, _ = self.run_step("encrypt")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Unsafe object path", result.stderr)
            self.assertFalse(any("get-object" in args for args in self.calls("aws")))
            self.assert_clean(result)

    def test_read_or_encryption_failure_is_redacted_and_removes_partial_output(self):
        for env in ({"FAIL_AWS": "list-objects-v2"}, {"FAIL_AWS": "get-object"}, {"FAIL_GPG": "1"}):
            with self.subTest(env=env):
                result, _ = self.run_step("encrypt", **env)
                self.assertNotEqual(result.returncode, 0)
                self.assert_clean(result)

    def test_upload_is_only_encrypted_file_with_one_day_retention(self):
        uploads = [s for s in self.job["steps"] if s.get("uses", "").startswith("actions/upload-artifact")]
        self.assertEqual(len(uploads), 1)
        upload = uploads[0]
        self.assertEqual(upload["uses"], "actions/upload-artifact@v4")
        self.assertEqual(upload["with"]["path"], ".run-export-artifact/run-inputs.tar.gpg")
        self.assertEqual(upload["with"]["retention-days"], 1)
        self.assertEqual(upload["with"]["if-no-files-found"], "error")
        self.assertNotIn("if", upload)  # Default success gating, never always().
        self.assertEqual(self.job["steps"][-1]["if"], "always()")
        self.assertLess(self.job["steps"].index(self.steps["encrypt"]), self.job["steps"].index(upload))

    @unittest.skipUnless(shutil.which("gpg"), "GnuPG not installed")
    def test_committed_public_key_encrypts_with_real_gpg_in_isolated_home(self):
        (self.tmp / "bin/gpg").unlink()
        result, _ = self.run_step("encrypt")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assert_clean(result, success=True)
        encrypted = (self.tmp / ".run-export-artifact/run-inputs.tar.gpg").read_bytes()
        self.assertNotIn(PAYLOAD.encode(), encrypted)
        self.assertFalse(tarfile.is_tarfile(self.tmp / ".run-export-artifact/run-inputs.tar.gpg"))


if __name__ == "__main__":
    unittest.main()
