"""The licence scan (scripts/licence_scan.py): the repository is clean, and each rule catches
what it is for. Every marker below is built by concatenation, so this file holds none."""

import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("licence_scan", ROOT / "scripts" / "licence_scan.py")
assert spec is not None and spec.loader is not None
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)

INRIA = "In" + "ria"
H3 = "H3" + "DGS"
CITY = "City" + "Gaussian"
MARKERS = {
    "nc-code-marker": "x = Cluster" + "Merger()\n",
    "nc-derivation": "# trans" + "cribed from " + INRIA + "'s code\n",
    "nc-project-in-source": "# the " + H3 + " way\n",
    "nc-model-id": 'MODEL = "black-forest-' + 'labs/FLUX.1-Fill-dev"\n',
    "nc-licence-in-source": 'licence = "CC BY-' + 'NC 4.0"\n',
    "withdrawn-asset": "ion = 14" + "15196\n",
}


def write(root: Path, path: str, text: str) -> str:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return path


def rules_of(findings) -> set:
    return {f.rule for f in findings}


def test_the_repository_is_clean() -> None:
    assert scan.main(["--root", str(ROOT)]) == 0


def test_each_rule_catches_its_marker(tmp_path: Path) -> None:
    for rule, text in MARKERS.items():
        path = write(tmp_path, f"src/{rule}.py", text)
        found = scan.scan(tmp_path, [path])
        assert rules_of(found) >= {rule}, rule
        assert all(f.path == path and f.line == 1 for f in found)
    notice = write(tmp_path, "pkg/NOTICE", "Notices\n\nmissing.py\n----------\nText.\n")
    found = scan.check_notices(tmp_path, [notice])
    assert [(f.rule, f.text) for f in found] == [("notice-names-missing-file", "missing.py")]


def test_a_derivation_verb_needs_a_non_commercial_name(tmp_path: Path) -> None:
    clean = write(tmp_path, "a.py", "# trans" + "cribed from gsplat's exporter\n")
    flagged = write(tmp_path, "b.py", "# trans" + "cribed from gsplat and " + INRIA + "\n")
    assert scan.scan(tmp_path, [clean]) == []
    assert "nc-derivation" in rules_of(scan.scan(tmp_path, [flagged]))


def test_a_paper_citation_with_an_arxiv_id_may_name_a_project_in_source(tmp_path: Path) -> None:
    cited = write(tmp_path, "a.py", "# the frozen ring of " + H3 + " (arXiv 2406.12080)\n")
    bare = write(tmp_path, "b.py", "# the frozen ring of " + H3 + "\n")
    assert scan.scan(tmp_path, [cited]) == []
    assert rules_of(scan.scan(tmp_path, [bare])) == {"nc-project-in-source"}


def test_prose_may_name_projects_but_not_code_markers(tmp_path: Path) -> None:
    prose = write(
        tmp_path, "notes.md", f"{CITY} and {H3} are non" + "commercial; CC BY-" + "NC 4.0.\n"
    )
    assert scan.scan(tmp_path, [prose]) == []
    marker = write(tmp_path, "more.md", "See Cluster" + "Merger.\n")
    assert rules_of(scan.scan(tmp_path, [marker])) == {"nc-code-marker"}
    docs = write(tmp_path, "docs/a.json", '{"a": "' + CITY + '"}\n')
    assert scan.scan(tmp_path, [docs]) == []


def _tree(tmp_path: Path, entries: list) -> Path:
    write(tmp_path, "src/a.py", MARKERS["withdrawn-asset"])
    allowlist = tmp_path / "allow.json"
    allowlist.write_text(json.dumps({"version": 1, "entries": entries}), encoding="utf-8")
    return allowlist


def test_an_allowlist_entry_that_matches_nothing_is_stale(tmp_path: Path) -> None:
    good = {"rule": "withdrawn-asset", "path": "src/a.py", "contains": "ion", "reason": "kept"}
    allowlist = _tree(tmp_path, [good])
    assert scan.main(["--root", str(tmp_path), "--allowlist", str(allowlist)]) == 0
    stale = {**good, "path": "src/gone.py"}
    allowlist = _tree(tmp_path, [good, stale])
    assert scan.main(["--root", str(tmp_path), "--allowlist", str(allowlist)]) == 2
    allowlist = _tree(tmp_path, [])
    assert scan.main(["--root", str(tmp_path), "--allowlist", str(allowlist)]) == 1


def test_an_allowlist_entry_needs_a_reason(tmp_path: Path) -> None:
    entry = {"rule": "withdrawn-asset", "path": "src/a.py", "contains": "ion", "reason": " "}
    allowlist = _tree(tmp_path, [entry])
    assert scan.main(["--root", str(tmp_path), "--allowlist", str(allowlist)]) == 2
    unknown = {**entry, "reason": "kept", "rule": "no-such-rule"}
    allowlist = _tree(tmp_path, [unknown])
    assert scan.main(["--root", str(tmp_path), "--allowlist", str(allowlist)]) == 2


def test_a_notice_must_name_files_that_exist(tmp_path: Path) -> None:
    write(tmp_path, "pkg/present.py", "x = 1\n")
    notice = write(
        tmp_path,
        "pkg/NOTICE",
        "Notices\n\npresent.py\n----------\nOurs.\n\nabsent.py\n---\nGone.\n\nNot a file\n---\n",
    )
    found = scan.check_notices(tmp_path, [notice])
    assert [(f.text, f.line) for f in found] == [("absent.py", 7)]


def test_generated_and_binary_paths_are_skipped(tmp_path: Path) -> None:
    text = MARKERS["nc-code-marker"]
    paths = [
        write(tmp_path, "data/tiles/x/tileset.json", text),
        write(tmp_path, "hardware/arm/viewer/data/meshes.json", text),
        write(tmp_path, "packages/world/fixtures/a.json", text),
    ]
    (tmp_path / "blob.bin").write_bytes(b"\0" + text.encode())
    (tmp_path / "latin.py").write_bytes(b"\xff" + text.encode())
    (tmp_path / "big.py").write_bytes(text.encode() * (scan.MAX_BYTES // len(text) + 1))
    paths += ["blob.bin", "latin.py", "big.py", *sorted(scan.SELF)]
    for own in scan.SELF:
        write(tmp_path, own, text)
    assert scan.scan(tmp_path, paths) == []
    assert scan.scan(tmp_path, [write(tmp_path, "src/b.py", text)]) != []


def test_lockfiles_and_licence_files_are_read(tmp_path: Path) -> None:
    """Where a dependency's or an asset's terms arrive as text: both lockfiles, and licence
    files with no extension, generated-data directories included."""
    licence = MARKERS["nc-licence-in-source"]
    paths = [
        write(tmp_path, "pnpm-lock.yaml", licence),
        write(tmp_path, "tools/x/uv.lock", licence),
        write(tmp_path, "vendor/thing/LICENSE", licence),
        write(tmp_path, "vendor/thing/COPYING", licence),
        write(tmp_path, "vendor/thing/LICENCE.txt", licence),
        write(tmp_path, "data/tiles/site/NOTICE", licence),
        write(tmp_path, "docs/assets/LICENSE.md", licence),
    ]
    for path in paths:
        assert rules_of(scan.scan(tmp_path, [path])) == {"nc-licence-in-source"}, path
    prose = write(tmp_path, "vendor/thing/README", licence)
    assert scan.scan(tmp_path, [prose]) == []


def test_a_tree_without_git_is_walked(tmp_path: Path) -> None:
    write(tmp_path, "src/a.py", "x = 1\n")
    write(tmp_path, "node_modules/pkg/index.js", MARKERS["nc-code-marker"])
    write(tmp_path, ".claude/notes.md", MARKERS["nc-code-marker"])
    done = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse"], capture_output=True, check=False
    )
    assert done.returncode != 0  # not inside a repository
    assert scan.tracked_files(tmp_path) == ["src/a.py"]
    assert scan.main(["--root", str(tmp_path)]) == 0
