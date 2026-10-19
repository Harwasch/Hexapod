@AGENTS.md

<!-- claude-ops pointer: keep everything below this block Claude-specific -->

## Working model

This repo is run by one coordinator thread. Coordinators follow the claude-ops playbook:
`/ops:orchestrate` (plugin `ops`, marketplace `Harwasch/claude-ops`).
In a cloud session the plugin is not loaded; read it at
https://github.com/Harwasch/claude-ops/blob/main/plugins/ops/skills/orchestrate/SKILL.md
The coordinator's log is `.claude/ops/LOG.md` (git-ignored, in the main checkout).

## Claude only

- CesiumJS skills live in `.claude/skills/` (mirrored for other agents in `.agents/skills/`); start from
  `using-cesiumjs-skills`.
- `.claude/settings.json` enables the `makehardware` and `kicad-happy` plugins for the hardware work.
- GitHub: the repo is owned by Harwasch; use `GH_TOKEN=$(gh auth token --user Harwasch)` for `gh`.
