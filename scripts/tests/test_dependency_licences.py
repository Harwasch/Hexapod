"""The dependency licence check (scripts/dependency_licences.py): which licences fail, the
allowlist, and the CLI's exit codes. Each non-commercial licence below is built by
concatenation, so the source scan finds none in this file."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "dependency_licences", ROOT / "scripts" / "dependency_licences.py"
)
assert spec is not None and spec.loader is not None
deps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deps)

NC = "N" + "C"
NON = "Non" + "-Commercial"


@pytest.mark.parametrize(
    "expression",
    [
        f"CC-BY-{NC}-4.0",
        f"CC-BY-{NC}-SA-4.0",
        f"CC-BY-{NC}-ND-4.0",
        f"MIT OR CC-BY-{NC}-4.0",
        "PolyForm-Non" + "commercial-1.0.0",
        f"LicenseRef-Acme-{NC}",
        f"NSCL-{NC}",
        f"{NON} Use Only",
        "For research use only",
        "Not for commercial use",
        "Commercial use is prohibited",
    ],
)
def test_a_licence_that_forbids_commercial_use_fails(expression: str) -> None:
    assert deps.problem_of(expression) == "non-commercial"


def test_a_non_commercial_classifier_or_free_text_fails() -> None:
    classifier = f"License :: Other/Proprietary License :: {NON}"
    assert deps.problem_of(None, [classifier], None) == "non-commercial"
    text = "Copyright 2024 Someone\n\nPermission is granted for " + "non" + "commercial use only."
    assert deps.problem_of(None, [], text) == "non-commercial"
    url = "https://creativecommons.org/licenses/by-" + "nc/4.0/"
    assert deps.problem_of(None, [], url) == "non-commercial"


@pytest.mark.parametrize(
    ("expression", "classifiers", "text"),
    [
        ("MIT", [], None),
        ("BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0", [], None),
        ("(MPL-2.0 OR Apache-2.0)", [], None),
        ("NCSA", [], None),
        ("CC-BY-4.0", [], None),
        (None, ["License :: OSI Approved :: MIT License"], None),
        (None, [], "Apache-2.0"),
        # The Unlicense and SSLeay permit both kinds of use.
        (None, [], "Free for any purpose, commercial or " + "non" + "-commercial.\nMore."),
        # The GPL's option for passing on object code, which numpy's and scipy's wheels carry.
        (None, [], "BSD\n\nThis alternative is allowed only occasionally and non" + "commercially"),
    ],
)
def test_a_permissive_licence_passes(
    expression: str | None, classifiers: list[str], text: str | None
) -> None:
    assert deps.problem_of(expression, classifiers, text) == ""


@pytest.mark.parametrize("text", [None, "", "UNKNOWN", "Unknown"])
def test_a_package_that_declares_no_licence_fails(text: str | None) -> None:
    assert deps.problem_of(None, [], text) == "undeclared"
    assert deps.problem_of(None, ["License :: OSI Approved"], text) == "undeclared"


def test_installed_metadata_rows_become_packages() -> None:
    rows = [
        {"name": "fine", "version": "1", "expression": "MIT", "license": None, "classifiers": []},
        {
            "name": "Model_Weights",
            "version": "2",
            "expression": None,
            "license": f"CC-BY-{NC}-4.0",
            "classifiers": [],
        },
        {"name": "ours", "version": "0", "expression": None, "license": None, "classifiers": []},
    ]
    found = deps.packages_of_metadata(rows, "tools/x", skip=["Ours"])
    assert [(p.name, p.problem) for p in found] == [
        ("fine", ""),
        ("Model_Weights", "non-commercial"),
    ]


def _listing(tmp_path: Path, listing: dict) -> Path:
    path = tmp_path / "pnpm.json"
    path.write_text(json.dumps(listing), encoding="utf-8")
    return path


def _allowlist(tmp_path: Path, entries: list) -> Path:
    path = tmp_path / "allow.json"
    path.write_text(json.dumps({"version": 1, "entries": entries}), encoding="utf-8")
    return path


LISTING = {
    "MIT": [{"name": "left-pad", "versions": ["1.3.0"]}],
    f"CC-BY-{NC}-SA-4.0": [{"name": "pretty-shapes", "versions": ["2.0.0"]}],
    "Unknown": [{"name": "mystery", "versions": ["0.1.0"]}],
}


def test_pnpm_packages_are_checked_by_their_licence(tmp_path: Path) -> None:
    found = deps.pnpm_packages(LISTING)
    assert {p.name: p.problem for p in found} == {
        "left-pad": "",
        "pretty-shapes": "non-commercial",
        "mystery": "undeclared",
    }
    listing = _listing(tmp_path, LISTING)
    empty = _allowlist(tmp_path, [])
    assert deps.main(["--pnpm-json", str(listing), "--allowlist", str(empty)]) == 1


def test_a_reviewed_exception_needs_its_licence_and_a_reason(tmp_path: Path) -> None:
    listing = _listing(tmp_path, LISTING)
    entries = [
        {"ecosystem": "npm", "name": "pretty-shapes", "licence": f"BY-{NC}-SA", "reason": "dev"},
        {"ecosystem": "npm", "name": "mystery", "licence": "Unknown", "reason": "read its file"},
    ]
    allow = _allowlist(tmp_path, entries)
    assert deps.main(["--pnpm-json", str(listing), "--allowlist", str(allow)]) == 0
    wrong = _allowlist(tmp_path, [{**entries[0], "licence": "MIT"}, entries[1]])
    assert deps.main(["--pnpm-json", str(listing), "--allowlist", str(wrong)]) == 2  # stale
    blank = _allowlist(tmp_path, [{**entries[0], "reason": " "}])
    assert deps.main(["--pnpm-json", str(listing), "--allowlist", str(blank)]) == 2
    other = _allowlist(tmp_path, [{**entries[0], "ecosystem": "cargo"}])
    assert deps.main(["--pnpm-json", str(listing), "--allowlist", str(other)]) == 2


def test_a_python_entry_is_stale_only_after_every_project_was_read(tmp_path: Path) -> None:
    listing = _listing(tmp_path, {"MIT": [{"name": "left-pad", "versions": ["1"]}]})
    entry = {"ecosystem": "python", "name": "gone", "licence": "X", "reason": "was reviewed"}
    allow = _allowlist(tmp_path, [entry])
    # One CI job reads one environment, so an entry for another one is not stale there.
    assert deps.main(["--pnpm-json", str(listing), "--allowlist", str(allow)]) == 0


def test_an_environment_that_cannot_be_read_is_an_error(tmp_path: Path) -> None:
    assert deps.main(["--root", str(tmp_path), "--project", "tools/none"]) == 2
    assert deps.main(["--python", str(tmp_path / "no-python")]) == 2


def test_uv_lock_names_are_read(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text(
        'version = 1\n\n[[package]]\nname = "Foo_Bar"\nversion = "1"\n\n'
        '[[package]]\nname = "baz"\nversion = "2"\n',
        encoding="utf-8",
    )
    assert deps.locked_names(lock) == {"foo-bar", "baz"}
    (tmp_path / "pyproject.toml").write_text(
        '[build-system]\nname = "no"\n\n[project]\nname = "twin-x"\nversion = "0"\n',
        encoding="utf-8",
    )
    assert deps.project_name(tmp_path.parent, tmp_path.name) == "twin-x"


def test_the_repository_allowlist_is_well_formed() -> None:
    for entry in deps.load_allowlist(ROOT / "scripts" / "dependency_licence_allowlist.json"):
        assert entry["reason"].strip()
