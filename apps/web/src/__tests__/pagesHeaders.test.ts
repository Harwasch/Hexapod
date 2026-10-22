/**
 * `infra/pages/_headers`, the response headers Cloudflare Pages sends: read the way Pages
 * applies them, so a test can say what one page actually gets.
 *
 * The case it exists for: the landing page shows the site finder (`finder.html`) in a frame,
 * and `X-Frame-Options: DENY` on every page made the browser refuse it ("refused to connect")
 * on the live site, while local tests, which never send these headers, passed.
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

const HEADERS = resolve(__dirname, "../../../../infra/pages/_headers");

interface Rule {
  path: string;
  set: [string, string][];
  detach: string[];
}

function rules(text: string): Rule[] {
  const found: Rule[] = [];
  for (const line of text.split("\n")) {
    if (line.trim() === "" || line.trimStart().startsWith("#")) continue;
    if (!/^\s/.test(line)) {
      found.push({ path: line.trim(), set: [], detach: [] });
      continue;
    }
    const rule = found.at(-1);
    if (!rule) throw new Error(`a header before any path: ${line}`);
    const entry = line.trim();
    if (entry.startsWith("!")) rule.detach.push(entry.slice(1).trim().toLowerCase());
    else {
      const colon = entry.indexOf(":");
      rule.set.push([entry.slice(0, colon).trim().toLowerCase(), entry.slice(colon + 1).trim()]);
    }
  }
  return found;
}

/** Pages' matching: an exact path, or a trailing `*` matching any rest. */
function matches(pattern: string, path: string): boolean {
  return pattern.endsWith("*") ? path.startsWith(pattern.slice(0, -1)) : pattern === path;
}

/** The headers Pages sends for `path`: every matching rule's, in order; `! Name` removes one. */
function headersFor(path: string): Map<string, string> {
  const applied = rules(readFileSync(HEADERS, "utf8")).filter((rule) => matches(rule.path, path));
  const headers = new Map<string, string>();
  for (const rule of applied)
    for (const [name, value] of rule.set)
      headers.set(name, headers.has(name) ? `${headers.get(name)}, ${value}` : value);
  for (const rule of applied) for (const name of rule.detach) headers.delete(name);
  return headers;
}

describe("Cloudflare Pages headers", () => {
  // Pages answers /finder.html with a redirect to /finder, so both are the finder.
  it.each(["/finder", "/finder.html"])("let %s be framed by this site alone", (path) => {
    const headers = headersFor(path);
    expect(headers.has("x-frame-options")).toBe(false);
    expect(headers.get("content-security-policy")).toBe("frame-ancestors 'self'");
    expect(headers.get("x-content-type-options")).toBe("nosniff");
  });

  it.each(["/", "/landing", "/landing.html", "/admin.html", "/upload.html", "/view.html"])(
    "keep %s out of every frame",
    (path) => {
      expect(headersFor(path).get("x-frame-options")).toBe("DENY");
    },
  );
});
