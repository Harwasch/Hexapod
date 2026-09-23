# Hardware workflow

This repository was created from the MakeHardware project template. The hardware parts use
the `makehardware` and `kicad-happy` Claude Code plugins declared in `.claude/settings.json`.

In the first hardware session run `/hw-new-project`, which scaffolds `plan.yaml`,
`requirements/`, `hw/`, `cad/`, `concepts/`, `sim/` and `strictdoc.toml`, then start the
vision interview with `Use hw-vision. I want to build <one sentence>.`

Commands used day to day:

```bash
hw-doctor                 # what the toolchain can actually do right now
/hw-status                # plan progress, what is ready to start, requirements coverage
plan-render               # refresh docs/plan.svg
block-diagram --check     # architecture gate; exit 1 on an over-budget rail
req-trace --gate          # traceability gate; exit 1 while gaps remain
```

The plugin skills need the toolchain described in
[MakeHardware's env/](https://github.com/Harwasch/MakeHardware/tree/HEAD/env) and its setup
script; see [docs/01-environment.md](https://github.com/Harwasch/MakeHardware/blob/HEAD/docs/01-environment.md).
