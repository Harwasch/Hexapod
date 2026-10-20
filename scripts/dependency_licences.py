#!/usr/bin/env python3
"""Dependency licences: no package the product installs may forbid commercial use.

The source scan (`scripts/licence_scan.py`) reads this repository's own files. This reads the
licences of what they install:

* **Python**: every distribution installed in a uv project's synced environment
  (`tools/pipeline`, `tools/captures`, `apps/api`), from its installed metadata
  (`importlib.metadata`: `License-Expression`, `License` and the `License ::` classifiers),
  read by that environment's own interpreter. A package `uv.lock` resolves that this platform
  did not install (a Windows-only shim, a group not synced) is counted as unchecked.
* **JavaScript**: `pnpm licenses list --json --prod --recursive`, built into pnpm 10: the
  production dependencies of every workspace package, from their installed `package.json`.

A licence fails when it forbids commercial use (a Creative Commons NC licence, PolyForm
Noncommercial, a custom identifier with an "NC" part, "non-commercial", "research use only",
"not for commercial use"), and when a package names no licence: none at all, a placeholder
("UNKNOWN", whether as `License-Expression` or as text, "none", "unlicensed"), or a pointer to
a file ("SEE LICENSE IN LICENSE"), any of which could be anything. Two phrases do not count:
"commercial or non-commercial" (the Unlicense, SSLeay: permission for both) and the GPL's
"occasionally and noncommercially" (a way of passing on object code, not a limit on use;
numpy's and scipy's wheels carry it for libgfortran).

Reviewed exceptions live in `scripts/dependency_licence_allowlist.json`: ecosystem, package
name, a substring of the licence as reported, why, and who read the licence (`checked_by`) on
which day (`checked_on`, YYYY-MM-DD). An entry that allows nothing is stale
and fails, but only in a run that read every environment of its ecosystem (a CI job has one
venv). Standard library only (Python 3.10+).

    python3 scripts/dependency_licences.py [--root DIR] [--project DIR]... [--python PY]...
        [--skip NAME]... [--pnpm | --pnpm-json FILE] [--allowlist FILE] [--json]

`--project` reads that uv project's `.venv` (leaving out the project's own install), `--python`
any interpreter's packages (`--skip NAME` leaves one out, such as the project the interpreter
belongs to). With none of these: every project in `PROJECTS` that has a `.venv`, and pnpm when
`node_modules` exists. Exit 0: clean. 1: a non-commercial, undeclared or see-file licence not
on the allowlist. 2: a stale or invalid allowlist entry, or an environment that cannot be read
(a pnpm listing that names no package counts: it cannot be told from a failed listing).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path
from typing import Any, NamedTuple

#: The uv projects whose environments build or run what the product ships.
PROJECTS = ("tools/pipeline", "tools/captures", "apps/api")
ECOSYSTEMS = ("python", "npm")
#: An allowlist entry's fields, all required.
KEYS = ("ecosystem", "name", "licence", "reason", "checked_by", "checked_on")

#: Wording of a licence that forbids commercial use; case-insensitive, any field.
NON_COMMERCIAL = re.compile(
    r"(?i)\bcc[-_ ]?by[-_ ]?nc|creativecommons\.org/licenses/by-nc|\bnon[-_ ]?commercial(?!ly)"
    r"|\bnot (?:be )?(?:used )?for commercial|\bno commercial use"
    r"|commercial (?:use|purposes?) (?:is |are )?(?:not permitted|not allowed|prohibited"
    r"|forbidden)|\bresearch (?:purposes |use )?only"
)
#: An identifier with an "NC" part ("LicenseRef-Acme-NC", "NSCL-NC"): case-sensitive, and
#: read only in the short fields, where "NC" cannot be a word of prose.
NC_TOKEN = re.compile(r"(?<![A-Za-z0-9])NC(?![A-Za-z])")
#: Wording that names non-commercial use to permit it alongside commercial use.
PERMISSIVE = re.compile(r"(?i)\bcommercial (?:and|or) non-?commercial")
#: A free-text `License` this long or shorter, on one line, is an identifier, not a text.
SHORT = 100
#: Placeholders that name no licence, compared lower-cased.
UNDECLARED = frozenset({"", "unknown", "none", "unlicensed", "unlicenced", "n/a"})
#: A pointer to a file instead of a licence ("SEE LICENSE IN LICENSE", npm's convention).
SEE_FILE = re.compile(r"(?i)^see licen[cs]e\b")
#: Classifiers that say nothing about which licence.
VACUOUS = frozenset({"License ::", "License :: OSI Approved"})

#: Run by each environment's interpreter, with -I, so nothing in the working directory loads.
_METADATA = """
import importlib.metadata as m, json, sys
json.dump([{
    "name": d.metadata["Name"], "version": d.version,
    "expression": d.metadata.get("License-Expression"),
    "license": d.metadata.get("License"),
    "classifiers": [c for c in (d.metadata.get_all("Classifier") or [])
                    if c.startswith("License ::")],
} for d in m.distributions()], sys.stdout)
"""


class Package(NamedTuple):
    ecosystem: str
    name: str
    version: str
    #: The licence as reported (expression, classifiers and free text, joined by " | ").
    licence: str
    #: Where it was found: a project, an interpreter or "pnpm".
    where: str
    #: "non-commercial", "undeclared", "see-file", or "" when the licence is fine.
    problem: str


class AllowlistError(ValueError):
    """An allowlist that cannot be read, or an entry that is not well formed."""


class UnreadableError(RuntimeError):
    """An environment that was asked for and could not be read."""


def normalise(name: str) -> str:
    """A package name as PEP 503 compares them (npm names just lower-cased)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def problem_of(
    expression: str | None, classifiers: Sequence[str] = (), text: str | None = None
) -> str:
    """ "non-commercial", "undeclared", "see-file" or "" for one package's licence fields: an
    SPDX-style expression, `License ::` classifiers and a free text (which may be a whole
    licence). A licence is named only by a field that is neither a placeholder nor a pointer
    to a file; with none, the package is "see-file" if some field points to a file, else
    "undeclared"."""
    free = (text or "").strip()
    short = [field for field in [(expression or "").strip(), *classifiers] if field]
    if free and "\n" not in free and len(free) <= SHORT:
        short.append(free)
    if any(NON_COMMERCIAL.search(PERMISSIVE.sub(" ", field)) for field in [*short, free]):
        return "non-commercial"
    if any(NC_TOKEN.search(field) for field in short):
        return "non-commercial"
    fields = [
        (expression or "").strip(),
        free,
        *(c for c in classifiers if c.strip() not in VACUOUS),
    ]
    if any(f.lower() not in UNDECLARED and not SEE_FILE.match(f) for f in fields):
        return ""
    return "see-file" if any(SEE_FILE.match(f) for f in fields) else "undeclared"


def project_name(root: Path, project: str) -> str | None:
    """The `[project] name` of a uv project, so its own (unlicensed) install is left out."""
    try:
        text = (root / project / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return None
    section = re.search(r"(?ms)^\[project\]\s*$(.*?)(?=^\[|\Z)", text)
    found = re.search(r'(?m)^name\s*=\s*"([^"]+)"', section.group(1)) if section else None
    return found.group(1) if found else None


def locked_names(lock: Path) -> set[str]:
    """The package names `uv.lock` resolves (its `[[package]]` tables)."""
    try:
        text = lock.read_text(encoding="utf-8")
    except OSError:
        return set()
    names = re.findall(r'(?m)^\[\[package\]\]\s*\nname = "([^"]+)"', text)
    return {normalise(name) for name in names}


def python_packages(python: str, where: str, skip: Iterable[str] = ()) -> list[Package]:
    """Every distribution installed for interpreter `python`."""
    try:
        done = subprocess.run(
            [python, "-I", "-c", _METADATA], capture_output=True, check=True, text=True
        )
        rows = json.loads(done.stdout)
    except (OSError, subprocess.CalledProcessError, ValueError) as problem:
        raise UnreadableError(f"{where}: cannot read installed metadata ({problem})") from None
    return packages_of_metadata(rows, where, skip)


def packages_of_metadata(
    rows: Iterable[dict[str, Any]], where: str, skip: Iterable[str] = ()
) -> list[Package]:
    """Packages of `_METADATA`'s rows (name, version, expression, license, classifiers)."""
    skipped = {normalise(name) for name in skip}
    out: list[Package] = []
    for row in rows:
        if normalise(row["name"]) in skipped:
            continue
        expression, text = row.get("expression"), row.get("license")
        classifiers = list(row.get("classifiers") or [])
        shown = [part for part in [expression or "", *classifiers, text or ""] if part.strip()]
        out.append(
            Package(
                "python",
                row["name"],
                row["version"],
                " | ".join(shown),
                where,
                problem_of(expression, classifiers, text),
            )
        )
    return out


def pnpm_packages(listing: dict[str, Any]) -> list[Package]:
    """Packages of a `pnpm licenses list --json` document (licence -> packages)."""
    out: list[Package] = []
    for licence, packages in listing.items():
        for package in packages:
            versions = package.get("versions") or [package.get("version", "")]
            out.append(
                Package(
                    "npm",
                    package["name"],
                    ",".join(versions),
                    licence,
                    "pnpm",
                    problem_of(None, (), licence),
                )
            )
    return out


def run_pnpm(root: Path) -> dict[str, Any]:
    """`pnpm licenses list --json --prod --recursive` in `root`, parsed."""
    pnpm = shutil.which("pnpm")
    corepack = shutil.which("corepack")
    command = [pnpm] if pnpm else [corepack, "pnpm"] if corepack else None
    if command is None:
        raise UnreadableError("pnpm: neither pnpm nor corepack is on PATH")
    try:
        done = subprocess.run(
            [*command, "licenses", "list", "--json", "--prod", "--recursive"],
            cwd=root,
            capture_output=True,
            check=True,
            text=True,
        )
        listing: dict[str, Any] = json.loads(done.stdout)
    except (OSError, subprocess.CalledProcessError, ValueError) as problem:
        raise UnreadableError(f"pnpm: cannot list licences ({problem})") from None
    return listing


def load_allowlist(path: Path) -> list[dict[str, str]]:
    """The entries, each checked: a known ecosystem, a package name, a substring of its
    reported licence, a reason and who checked it, all non-empty strings, and the day they
    checked it as YYYY-MM-DD."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as problem:
        raise AllowlistError(f"{path}: unreadable ({problem})") from None
    if not isinstance(document, dict) or document.get("version") != 1:
        raise AllowlistError(f"{path}: expected {{'version': 1, 'entries': [...]}}")
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise AllowlistError(f"{path}: 'entries' must be a list")
    out: list[dict[str, str]] = []
    for number, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise AllowlistError(f"{path}: entry {number} is not an object")
        for key in KEYS:
            value = entry.get(key)
            if not isinstance(value, str) or not value.strip():
                raise AllowlistError(f"{path}: entry {number} needs a non-empty {key!r}")
        if entry["ecosystem"] not in ECOSYSTEMS:
            raise AllowlistError(
                f"{path}: entry {number} names no ecosystem: {entry['ecosystem']!r}"
            )
        try:
            date.fromisoformat(entry["checked_on"])
        except ValueError:
            raise AllowlistError(
                f"{path}: entry {number} needs 'checked_on' as YYYY-MM-DD, not "
                f"{entry['checked_on']!r}"
            ) from None
        out.append({key: str(entry[key]) for key in KEYS})
    return out


def _entry_of(package: Package, entries: Sequence[dict[str, str]]) -> int | None:
    for index, entry in enumerate(entries):
        if (
            entry["ecosystem"] == package.ecosystem
            and normalise(entry["name"]) == normalise(package.name)
            and entry["licence"] in package.licence
        ):
            return index
    return None


def apply_allowlist(
    packages: Sequence[Package], entries: Sequence[dict[str, str]]
) -> tuple[list[Package], list[Package], set[int]]:
    """(problem packages no entry allows, those an entry allows, the entries used)."""
    used: set[int] = set()
    left: list[Package] = []
    allowed: list[Package] = []
    for package in packages:
        if not package.problem:
            continue
        index = _entry_of(package, entries)
        if index is None:
            left.append(package)
        else:
            used.add(index)
            allowed.append(package)
    return left, allowed, used


def project_packages(root: Path, project: str) -> tuple[list[Package], str]:
    """The packages installed in uv project `project`'s .venv, less the project itself, and
    a note of how many locked packages this platform did not install."""
    venv = root / project / ".venv" / "bin" / "python"
    if not venv.exists():
        raise UnreadableError(f"{project}: no .venv (run `uv sync` there)")
    own = [name for name in [project_name(root, project)] if name]
    found = python_packages(str(venv), project, skip=own)
    locked = locked_names(root / project / "uv.lock") - {normalise(n) for n in own}
    unchecked = locked - {normalise(p.name) for p in found}
    note = (
        f"{project}: {len(found)} installed; {len(unchecked)} locked for other platforms or "
        "groups, not installed here, unchecked"
    )
    return found, note


def collect(
    root: Path,
    projects: Sequence[str],
    pythons: Sequence[str],
    pnpm: bool,
    pnpm_json: Path | None,
    skip: Sequence[str] = (),
) -> tuple[list[Package], list[str], dict[str, bool]]:
    """(packages, notes, ecosystem -> read in full: every project in `PROJECTS`, or a pnpm
    listing). With no option: every project that has a .venv, and pnpm when node_modules
    exists."""
    explicit = bool(projects or pythons or pnpm or pnpm_json)
    packages: list[Package] = []
    notes: list[str] = []
    read: set[str] = set()
    for python in pythons:
        packages += python_packages(python, python, skip=skip)
    for project in projects if explicit else PROJECTS:
        if not explicit and not (root / project / ".venv").exists():
            notes.append(f"{project}: no .venv (run `uv sync`), not checked")
            continue
        found, note = project_packages(root, project)
        packages += found
        notes.append(note)
        read.add(project)
    listing: dict[str, Any] | None = None
    if pnpm_json is not None:
        listing = json.loads(pnpm_json.read_text(encoding="utf-8"))
    elif pnpm or (not explicit and (root / "node_modules").is_dir()):
        listing = run_pnpm(root)
    elif not explicit:
        notes.append("pnpm: no node_modules (run `pnpm install`), not checked")
    if listing is not None:
        found = pnpm_packages(listing)
        if not found:
            raise UnreadableError("pnpm: the licence listing names no package")
        packages += found
        notes.append(f"pnpm: {len(found)} production packages")
    complete = {"python": read >= set(PROJECTS), "npm": listing is not None}
    return packages, notes, complete


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--project", action="append", default=[], help="a uv project whose .venv to read"
    )
    parser.add_argument(
        "--python", action="append", default=[], help="an interpreter whose packages to read"
    )
    parser.add_argument(
        "--skip", action="append", default=[], help="a package name --python leaves out"
    )
    parser.add_argument("--pnpm", action="store_true", help="read pnpm's production packages")
    parser.add_argument("--pnpm-json", type=Path, help="a saved `pnpm licenses list --json`")
    parser.add_argument(
        "--allowlist", type=Path, help="default: ROOT/scripts/dependency_licence_allowlist.json"
    )
    parser.add_argument("--json", action="store_true", help="the result as JSON")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        packages, notes, complete = collect(
            root, args.project, args.python, args.pnpm, args.pnpm_json, args.skip
        )
    except (UnreadableError, OSError, ValueError) as problem:
        sys.stderr.write(f"dependency_licences: {problem}\n")
        return 2
    allowlist = args.allowlist or root / "scripts" / "dependency_licence_allowlist.json"
    try:
        entries = load_allowlist(allowlist) if args.allowlist or allowlist.is_file() else []
    except AllowlistError as problem:
        sys.stderr.write(f"dependency_licences: {problem}\n")
        return 2
    left, allowed, used = apply_allowlist(packages, entries)
    stale = [
        entry
        for index, entry in enumerate(entries)
        if index not in used and complete[entry["ecosystem"]]
    ]
    if args.json:
        document = {
            "findings": [p._asdict() for p in left],
            "allowlisted": [p._asdict() for p in allowed],
            "stale": stale,
            "notes": notes,
            "packages": len(packages),
        }
        sys.stdout.write(json.dumps(document, indent=1) + "\n")
    else:
        for p in left:
            sys.stdout.write(
                f"{p.problem} {p.ecosystem} {p.name} {p.version} ({p.where}): "
                f"{p.licence[:200] or '(none declared)'}\n"
            )
        for entry in stale:
            sys.stdout.write(f"STALE {entry['ecosystem']} {entry['name']}: allows nothing\n")
        for note in notes:
            sys.stdout.write(f"  {note}\n")
        sys.stdout.write(
            f"{len(left)} findings, {len(allowed)} allowlisted, {len(stale)} stale "
            f"({len(packages)} packages)\n"
        )
    if stale:
        return 2
    return 1 if left else 0


if __name__ == "__main__":
    raise SystemExit(main())
