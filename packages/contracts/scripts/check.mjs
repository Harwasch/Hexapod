// Fails when the committed generated types are stale relative to openapi.json.
import { readFileSync, unlinkSync } from "node:fs";
import { resolve } from "node:path";

const dir = resolve(import.meta.dirname, "../src/generated");
const committed = readFileSync(resolve(dir, "api.d.ts"), "utf8");
const fresh = readFileSync(resolve(dir, "api.check.d.ts"), "utf8");
unlinkSync(resolve(dir, "api.check.d.ts"));
if (committed !== fresh) {
  console.error(
    "packages/contracts/src/generated/api.d.ts is out of date. Run `pnpm contracts:generate`.",
  );
  process.exit(1);
}
console.info("contracts: generated types are up to date");
