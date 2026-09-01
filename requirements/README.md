# Requirements

Plain-text StrictDoc requirements in git. They diff and review like code.

Shared grammar: [`hardware.sgra`](hardware.sgra) — the `.sgra` format does not
allow comments, so the fields are documented here.

## Fields

| Field | Required | Meaning |
|---|---|---|
| `UID` | yes | `<PREFIX>-<number>`. The prefix carries the level (below). |
| `TITLE` | yes | Short label. |
| `STATEMENT` | yes | One "shall", with a number and a unit wherever physics allows one. |
| `RATIONALE` | no | **Why this number.** What you re-read when the number has to move. |
| `VERIFICATION` | yes | `Analysis` \| `Simulation` \| `Test` \| `Inspection`. |
| `EVIDENCE` | no | Where the proof lives. Empty means unverified, and the gate counts it. |
| `BUDGET` | no | The share of a parent budget this consumes, so roll-ups can be checked. |
| `STATUS` | yes | `Draft` \| `Agreed` \| `Implemented` \| `Verified` \| `Waived`. |

Relations: `Parent` with role `Refines` (decomposition), and `File`
(implementation traceability to the model, deck or sheet that realises it).

## Levels

| Prefix | Level | Testable? | Refines |
|---|---|---|---|
| `VIS-` | vision intent, in the human's words | no, deliberately | nothing |
| `SYS-` | system requirement | yes | a `VIS-` |
| `ELE-` | electrical | yes | a `SYS-` |
| `MEC-` | mechanical | yes | a `SYS-` |
| `FW-` | firmware | yes | a `SYS-` |
| `MFG-` | manufacturing / test | yes | a `SYS-` |

## Syntax that bites

Multi-line fields need explicit block markers, and bodies render as RST — a
bare `SYS-*` breaks the build on an unterminated emphasis span:

```
RATIONALE: >>>
Derived from SYS-001: a 900 mAh cell, 12 h active at 65 mA, and a 7-day
standby tail leaves 40 uA for the always-on rail. Write ``SYS-`` not SYS-*.
<<<
```

## Checking

```bash
# Referential integrity: duplicate UIDs, dangling parents. Fails the build.
/opt/hw-py/bin/strictdoc export requirements \
    --output-dir build/requirements --formats=html,json

# Decomposition quality: orphans, undecomposed levels, missing evidence.
/opt/hw-py/bin/python scripts/req_trace.py
/opt/hw-py/bin/python scripts/req_trace.py --gate    # exit 1 on any gap
```

The files here ship with one worked example per level. Replace them.
