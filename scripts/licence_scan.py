#!/usr/bin/env python3
"""Licence scan: no code, model or asset this project may not ship comes back unnoticed.

Looks through every tracked file (`git ls-files`; a walk of the tree when there is no git)
for the markers of code, models and assets whose licences forbid commercial use: the file
and symbol names of non-commercial reference implementations, a line saying code was taken
from one, a non-commercial project named in source (a paper citation with its arXiv id on
the same line is fine), a non-commercial model id, a non-commercial licence in source, and
the withdrawn ion sample mesh. Every NOTICE must name only files that exist beside it.

Licence files (LICENSE, LICENCE, COPYING, NOTICE, COPYRIGHT, with or without an extension)
are read as source wherever they are, generated-data directories included: that is where an
asset's terms arrive. The licences of installed packages are `scripts/dependency_licences.py`'s
job; this scan reads only the repository's own files (lockfiles included, as text).

A finding that is deliberate is listed in the allowlist (`scripts/licence_allowlist.json`):
rule, exact path, a substring of the line, and why. An entry that matches nothing is stale
and fails the scan, so the list cannot rot. Standard library only (Python 3.10+).

    python scripts/licence_scan.py [--root DIR] [--allowlist FILE] [--summary] [--json]

Exit 0: clean. 1: findings not on the allowlist. 2: a stale or invalid allowlist entry.
docs/DECISIONS/0012-licence-clean-room.md says why each rule exists.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, NamedTuple


class Rule(NamedTuple):
    id: str
    #: "all" files, or "source" only (code and config, not docs/ or Markdown).
    scope: str
    pattern: re.Pattern[str]
    #: A second pattern the same line must also match (None: the first is enough).
    also: re.Pattern[str] | None = None
    #: A pattern that, matched on the same line, clears it.
    exempt: re.Pattern[str] | None = None


class Finding(NamedTuple):
    rule: str
    path: str
    line: int
    text: str


RULES: tuple[Rule, ...] = (
    Rule(
        "nc-code-marker",
        "all",
        re.compile(
            r"ClusterMerger|hierarchy_explicit_loader|gaussian-hierarchy|hierarchy-rasterizer"
            r"|runtime_switching|ellipseSurface|matrixFromQuat|\bgetWeight\b|train_post\.py"
            r"|full_train\.py|post_opt_chunk_args|getNerfppNorm|get_expon_lr_func"
            r"|create_from_hier|citygs_partitioning_utils|build_partition_coordinates"
            r"|projection_based_partition_assignment|data_partition\.py|colmap_block_dataparser"
            r"|partition_citygs|sh_utils\.py|\beval_sh\b|loss_utils\.py"
            r"|diff-gaussian-rasterization|gaussian_model\.py|dataset_readers\.py|graphdeco"
        ),
    ),
    Rule(
        "nc-derivation",
        "all",
        re.compile(
            r"(?i)(transcri|verbatim|ported from|port of|copied from|adapted from|taken from"
            r"|lifted from)"
        ),
        also=re.compile(
            r"(?i)(inria|graphdeco|h3dgs|hierarchical[ -]?3dgs|gaussian-hierarchy|citygs"
            r"|citygaussian|\bcityg\b)"
        ),
    ),
    Rule(
        "nc-project-in-source",
        "source",
        re.compile(
            r"(?i)\bh3dgs|hierarchical[ -]3dgs|hierarchical 3d gaussian|citygaussian|\bcitygs\b"
            r"|\bcityg\b|\binria\b|graphdeco"
        ),
        exempt=re.compile(r"arXiv[: ]?\d{4}\.\d{4,5}"),
    ),
    Rule(
        "nc-model-id",
        "all",
        re.compile(
            r"FLUX\.1-[A-Za-z]+-dev|FLUX\.1-dev|FLUX\.1 \w+ \[dev\]|FluxFillPipeline"
            r"|black-forest-labs/"
        ),
    ),
    Rule(
        "nc-licence-in-source",
        "source",
        re.compile(r"(?i)\bcc[- ]?by[- ]?nc\b|creativecommons\.org/licenses/by-nc|non-?commercial"),
    ),
    Rule("withdrawn-asset", "all", re.compile(r"(?i)aerometrex|\b1415196\b")),
)
#: Not a line rule: checked by `check_notices`.
NOTICE_RULE = "notice-names-missing-file"
RULE_IDS = frozenset([*(rule.id for rule in RULES), NOTICE_RULE])

#: Skipped by the walk of a tree without git (git already leaves out what it ignores).
WALK_SKIP = frozenset(
    {
        ".git",
        "node_modules",
        ".venv",
        "dist",
        "coverage",
        "test-results",
        "playwright-report",
        "__pycache__",
        ".claude",
    }
)
#: Generated data: tiles, fixtures, a viewer's baked meshes.
SKIP_PREFIXES = ("data/tiles/", "hardware/arm/viewer/data/", "packages/world/fixtures/")
SKIP_NAMES: frozenset[str] = frozenset()
#: A licence file's name: read as source, and never skipped as generated data.
LICENCE_FILE = re.compile(r"(?i)^(licen[cs]e|copying|notice|copyright)([._-].*)?$")
#: The scan, its allowlist and its tests name every marker on purpose.
SELF = frozenset(
    {
        "scripts/licence_scan.py",
        "scripts/licence_allowlist.json",
        "scripts/tests/test_licence_scan.py",
        "tools/pipeline/tests/test_licence_scan.py",
        "scripts/dependency_licences.py",
        "scripts/dependency_licence_allowlist.json",
        "scripts/tests/test_dependency_licences.py",
        "apps/web/src/__tests__/dependencyLicences.test.ts",
    }
)
MAX_BYTES = 2 * 1024 * 1024
SOURCE_SUFFIXES = frozenset(
    {
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".mjs",
        ".cjs",
        ".glsl",
        ".wgsl",
        ".yaml",
        ".yml",
        ".toml",
        ".sh",
        ".json",
        ".patch",
        ".lock",
        ".html",
        ".css",
    }
)
_RULE_LINE = re.compile(r"^-{3,}\s*$")
_HAS_SUFFIX = re.compile(r"\.[A-Za-z0-9]+$")


class AllowlistError(ValueError):
    """An allowlist that cannot be read, or an entry that is not well formed."""


def tracked_files(root: Path) -> list[str]:
    """Paths relative to `root`, POSIX-style: what git tracks, else every file in a walk."""
    try:
        done = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            capture_output=True,
            check=True,
        )
        top = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            check=True,
            text=True,
        )
        if Path(top.stdout.strip()).resolve() == root.resolve():
            return sorted(p for p in done.stdout.decode("utf-8").split("\0") if p)
    except (OSError, subprocess.CalledProcessError):
        pass
    out: list[str] = []
    for directory, folders, names in os.walk(root):
        folders[:] = sorted(f for f in folders if f not in WALK_SKIP)
        for name in names:
            out.append(Path(directory, name).relative_to(root).as_posix())
    return sorted(out)


def is_licence_file(path: str) -> bool:
    """LICENSE, COPYING, NOTICE and the like, with any extension or none."""
    return LICENCE_FILE.match(path.rsplit("/", 1)[-1]) is not None


def _skipped(path: str) -> bool:
    if path in SELF:
        return True
    if is_licence_file(path):
        return False
    return path.startswith(SKIP_PREFIXES) or path.rsplit("/", 1)[-1] in SKIP_NAMES


def is_source(path: str) -> bool:
    """Code, config and licence files; Markdown and everything else under docs/ are prose."""
    if is_licence_file(path):
        return True
    return Path(path).suffix.lower() in SOURCE_SUFFIXES and not path.startswith("docs/")


def _text(root: Path, path: str) -> str | None:
    """The file as text, or None: missing, over `MAX_BYTES`, binary or not UTF-8."""
    full = root / path
    try:
        if not full.is_file() or full.stat().st_size > MAX_BYTES:
            return None
        raw = full.read_bytes()
    except OSError:
        return None
    if b"\0" in raw:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def scan(root: Path, files: Iterable[str], rules: Sequence[Rule] = RULES) -> list[Finding]:
    """Every (rule, line) a rule matches, one finding per rule per line."""
    out: list[Finding] = []
    for path in files:
        if _skipped(path):
            continue
        text = _text(root, path)
        if text is None:
            continue
        source = is_source(path)
        applicable = [rule for rule in rules if rule.scope == "all" or source]
        for number, line in enumerate(text.splitlines(), start=1):
            for rule in applicable:
                if not rule.pattern.search(line):
                    continue
                if rule.also is not None and not rule.also.search(line):
                    continue
                if rule.exempt is not None and rule.exempt.search(line):
                    continue
                out.append(Finding(rule.id, path, number, line.strip()))
    return out


def check_notices(root: Path, files: Iterable[str]) -> list[Finding]:
    """A NOTICE's section title (a line followed by a line of three or more '-') that ends
    in a file extension names a file that must exist beside the NOTICE."""
    out: list[Finding] = []
    for path in files:
        if path.rsplit("/", 1)[-1] != "NOTICE":
            continue
        text = _text(root, path)
        if text is None:
            continue
        lines = text.splitlines()
        for index in range(len(lines) - 1):
            title = lines[index].strip()
            if not title or not _RULE_LINE.match(lines[index + 1]):
                continue
            if _HAS_SUFFIX.search(title) and not (root / path).parent.joinpath(title).is_file():
                out.append(Finding(NOTICE_RULE, path, index + 1, title))
    return out


def load_allowlist(path: Path) -> list[dict[str, str]]:
    """The entries of an allowlist file, each checked: a known rule, an exact path, the
    substring its line must contain and a reason, all non-empty strings."""
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
        for key in ("rule", "path", "contains", "reason"):
            value = entry.get(key)
            if not isinstance(value, str) or not value.strip():
                raise AllowlistError(f"{path}: entry {number} needs a non-empty {key!r}")
        if entry["rule"] not in RULE_IDS:
            raise AllowlistError(f"{path}: entry {number} names no rule: {entry['rule']!r}")
        out.append({key: str(entry[key]) for key in ("rule", "path", "contains", "reason")})
    return out


def _entry_of(finding: Finding, entries: Sequence[dict[str, str]]) -> int | None:
    for index, entry in enumerate(entries):
        if (
            entry["rule"] == finding.rule
            and entry["path"] == finding.path
            and entry["contains"] in finding.text
        ):
            return index
    return None


def apply_allowlist(
    findings: Sequence[Finding], entries: Sequence[dict[str, str]]
) -> tuple[list[Finding], list[dict[str, str]]]:
    """(the findings no entry allows, the entries that allow nothing)."""
    used: set[int] = set()
    left: list[Finding] = []
    for finding in findings:
        index = _entry_of(finding, entries)
        if index is None:
            left.append(finding)
        else:
            used.add(index)
    stale = [entry for index, entry in enumerate(entries) if index not in used]
    return left, stale


def _short(text: str, limit: int = 200) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _default_allowlist(root: Path) -> Path | None:
    path = root / "scripts" / "licence_allowlist.json"
    return path if path.is_file() else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--allowlist", type=Path, default=None, help="default: ROOT/scripts/licence_allowlist.json"
    )
    parser.add_argument("--summary", action="store_true", help="allowlisted findings and counts")
    parser.add_argument("--json", action="store_true", help="the result as JSON")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    files = tracked_files(root)
    findings = scan(root, files) + check_notices(root, files)
    allowlist = args.allowlist or _default_allowlist(root)
    try:
        entries = load_allowlist(allowlist) if allowlist is not None else []
    except AllowlistError as problem:
        sys.stderr.write(f"licence_scan: {problem}\n")
        return 2
    left, stale = apply_allowlist(findings, entries)
    allowed = [f for f in findings if f not in left]
    if args.json:
        document: dict[str, Any] = {
            "findings": [f._asdict() for f in left],
            "allowlisted": [
                {**f._asdict(), "reason": entries[_entry_of(f, entries) or 0]["reason"]}
                for f in allowed
            ],
            "stale": stale,
        }
        sys.stdout.write(json.dumps(document, indent=1) + "\n")
    else:
        for f in left:
            sys.stdout.write(f"{f.rule} {f.path}:{f.line}: {_short(f.text)}\n")
        for entry in stale:
            sys.stdout.write(
                f"STALE {entry['rule']} {entry['path']}: no line contains {entry['contains']!r}\n"
            )
        if args.summary:
            for f in allowed:
                reason = entries[_entry_of(f, entries) or 0]["reason"]
                sys.stdout.write(
                    f"allowed {f.rule} {f.path}:{f.line}: {_short(f.text)} -- {reason}\n"
                )
            counts = Counter(f.rule for f in findings)
            for rule in sorted(RULE_IDS):
                sys.stdout.write(
                    f"  {rule}: {counts.get(rule, 0)} found, "
                    f"{sum(1 for f in allowed if f.rule == rule)} allowlisted\n"
                )
            sys.stdout.write(
                f"{len(left)} findings, {len(allowed)} allowlisted, {len(stale)} stale "
                f"({len(files)} files)\n"
            )
    if stale:
        return 2
    return 1 if left else 0


if __name__ == "__main__":
    raise SystemExit(main())
