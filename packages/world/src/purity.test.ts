import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

const SRC = dirname(fileURLToPath(import.meta.url));
const FILES = readdirSync(SRC).filter((name) => name.endsWith(".ts"));
const SELF = "purity.test.ts";

/** Strips block and line comments, so prose about a banned name is not mistaken for a use. */
function stripComments(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, " ").replace(/(^|[^:])\/\/.*$/gm, "$1");
}

function importsOf(source: string): string[] {
  const specifiers: string[] = [];
  const pattern =
    /(?:^|\s)(?:import|export)[\s\S]*?from\s+["']([^"']+)["']|\bimport\s*\(\s*["']([^"']+)["']/g;
  let match: RegExpExecArray | null = pattern.exec(source);
  while (match !== null) {
    const specifier = match[1] ?? match[2];
    if (specifier !== undefined) specifiers.push(specifier);
    match = pattern.exec(source);
  }
  return specifiers;
}

/** Test files may reach for the runner and Node's filesystem; the library may not. */
const TEST_ONLY_IMPORTS = new Set(["vitest", "node:fs", "node:path", "node:url"]);

const BANNED_GLOBALS: readonly [RegExp, string][] = [
  [/\bwindow\b/, "window"],
  [/\bdocument\b/, "document"],
  [/\bnavigator\b/, "navigator"],
  [/\bperformance\b/, "performance"],
  [/\brequestAnimationFrame\b/, "requestAnimationFrame"],
  [/\bHTML[A-Z]\w*Element\b/, "an HTML element type"],
  [/\bMath\.random\b/, "Math.random"],
  [/\bDate\.now\b/, "Date.now"],
  [/\bnew Date\b/, "new Date"],
  [/cesium/i, "Cesium"],
];

describe("package purity", () => {
  it("has sources to check", () => {
    expect(FILES.length).toBeGreaterThan(8);
    expect(FILES).toContain("index.ts");
  });

  it("imports nothing outside the package", () => {
    for (const name of FILES) {
      // This file quotes forbidden names on purpose, to prove the checker sees them.
      if (name === SELF) continue;
      const isTest = name.endsWith(".test.ts");
      const source = readFileSync(join(SRC, name), "utf8");
      for (const specifier of importsOf(source)) {
        if (specifier.startsWith("./") || specifier.startsWith("../")) continue;
        if (isTest && TEST_ONLY_IMPORTS.has(specifier)) continue;
        throw new Error(`${name} imports "${specifier}"; @twin/world must stay framework-free`);
      }
    }
  });

  it("names no Cesium, DOM or clock API", () => {
    for (const name of FILES) {
      if (name === SELF) continue;
      const code = stripComments(readFileSync(join(SRC, name), "utf8"));
      for (const [pattern, label] of BANNED_GLOBALS) {
        expect({ file: name, uses: pattern.test(code) ? label : null }).toEqual({
          file: name,
          uses: null,
        });
      }
    }
  });

  it("would catch a forbidden import if one appeared", () => {
    expect(importsOf('import { Cartesian3 } from "cesium";')).toEqual(["cesium"]);
    expect(importsOf('export * from "./vec";')).toEqual(["./vec"]);
    expect(importsOf('const m = await import("cesium");')).toEqual(["cesium"]);
    expect(stripComments("/** no Math.random here */\nconst a = 1;")).not.toMatch(/Math\.random/);
    expect(stripComments("const a = Math.random();")).toMatch(/Math\.random/);
  });
});
