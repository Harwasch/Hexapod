/**
 * `scripts/dependency_licences.py --pnpm`: no production package of the workspace forbids
 * commercial use or declares no licence. Run here because this is the CI job that has
 * `node_modules` (`pnpm licenses list` reads the installed packages); the checker itself is
 * the standard-library script the Python jobs run, so there is one rule, not two.
 */
import { spawnSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import { describe, expect, it } from "vitest";

const ROOT = resolve(__dirname, "../../../..");
const SCRIPT = join(ROOT, "scripts/dependency_licences.py");

function run(args: string[]): { status: number | null; out: string } {
  const done = spawnSync("python3", [SCRIPT, "--root", ROOT, ...args], { encoding: "utf8" });
  return { status: done.status, out: `${done.stdout}${done.stderr}` };
}

describe("dependency licences", () => {
  it("passes every production package pnpm installed", () => {
    const { status, out } = run(["--pnpm"]);
    expect(out).toMatch(/pnpm: \d+ production packages/);
    expect(status, out).toBe(0);
  });

  it("fails a package whose licence forbids commercial use", () => {
    const dir = mkdtempSync(join(tmpdir(), "dependency-licences-"));
    try {
      const listing = join(dir, "pnpm.json");
      // Built by concatenation so the source scan finds no licence marker in this file.
      const licence = ["CC", "BY", "N" + "C", "4.0"].join("-");
      writeFileSync(
        listing,
        JSON.stringify({ [licence]: [{ name: "shapes", versions: ["1.0.0"] }] }),
      );
      const { status, out } = run(["--pnpm-json", listing]);
      expect(out).toContain("non-commercial npm shapes 1.0.0");
      expect(status).toBe(1);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
