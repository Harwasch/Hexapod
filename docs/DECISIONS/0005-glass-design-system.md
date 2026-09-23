# ADR 0005 — A token-driven glass design system in `packages/ui`

**Status:** accepted · **Date:** 2026-09-15

## Context

The map must dominate; UI floats above it. Ad-hoc blurred panels would drift in radius,
blur, contrast and motion, and would be hard to keep accessible.

## Decision

All visual values are CSS custom properties (`--glass-*`, `--text-*`, `--accent`, motion
and spacing tokens) with light/dark, reduced-transparency, high-contrast and reduced-motion
variants driven by media queries and `data-*` attributes on `<html>`. Primitives
(`GlassPanel`, `GlassPill`, `GlassButton`, `GlassSegmentedControl`, `GlassPopover`,
`GlassSheet`, `GlassTooltip`, inputs, switch, slider) wrap Radix for keyboard and screen
reader behaviour. Glass CSS is written once in `packages/ui/src/styles/glass.css`.

## Consequences

- Consistent material across every surface; theming is a variable swap.
- Accessibility is built into the primitives (labels, focus rings, Escape/focus traps).
- Apps use the system font stack; no proprietary fonts or brand assets are bundled.
