/**
 * Search by meaning on real segmented scans, through the real API encoder. Skipped unless
 * both are given:
 *
 *   INSTANCES_MEANING_DIRS=/path/spool,/path/pumpkin,/path/camp   (instances.json + .emb)
 *   INSTANCES_MEANING_API=http://127.0.0.1:8000                   (TEXT_ENCODER_DIR set)
 *   INSTANCES_MEANING_QUERIES='spool=spool|cable spool;camp=cabin|flagpole'  (optional)
 *
 * It drives the store (`state/instances.ts`) as the panel does and prints each query's top
 * five, by tags alone and by tags and meaning.
 */
import { readFileSync } from "node:fs";
import { basename, join } from "node:path";

import { afterAll, describe, expect, it } from "vitest";

import type { QueryEmbedding } from "@/api/textEmbedding";
import {
  parseEmbeddings,
  parseInstances,
  searchInstances,
  subtreeSplats,
  type SearchResult,
} from "@/lib/instances";
import { configureMeaning, useInstances } from "@/state/instances";

const DIRS = (process.env.INSTANCES_MEANING_DIRS ?? "").split(",").filter(Boolean);
const API = process.env.INSTANCES_MEANING_API ?? "";
const QUERIES: Record<string, string[]> = {
  spool: ["spool", "cable spool"],
  pumpkin: ["pumpkin"],
  camp: ["cabin", "flagpole", "conifer"],
};
for (const part of (process.env.INSTANCES_MEANING_QUERIES ?? "").split(";").filter(Boolean)) {
  const [scan, list] = part.split("=");
  if (scan && list) QUERIES[scan] = list.split("|");
}

describe.skipIf(DIRS.length === 0 || !API)("search by meaning on real scans", () => {
  afterAll(() => configureMeaning(null));

  for (const dir of DIRS) {
    const scan = basename(dir);
    it(`${scan}: ${(QUERIES[scan] ?? []).join(", ")}`, async () => {
      const doc = parseInstances(JSON.parse(readFileSync(join(dir, "instances.json"), "utf8")));
      if (!doc?.embedding) throw new Error(`${dir}: no instances.json with an embedding`);
      const bytes = readFileSync(join(dir, doc.embedding.file));
      configureMeaning({
        debounceMs: 0,
        loadEmbeddings: (_url, ref) =>
          Promise.resolve(
            parseEmbeddings(
              bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.length),
              ref,
            ),
          ),
        encode: async (text): Promise<QueryEmbedding> => {
          const started = performance.now();
          const response = await fetch(
            `${API}/api/v1/text-embeddings?text=${encodeURIComponent(text)}`,
          );
          if (!response.ok) throw new Error(`encoder answered ${String(response.status)}`);
          const data = (await response.json()) as {
            model: string;
            prompt: string;
            embedding: number[];
          };
          console.info(`  [${text}] encoder ${(performance.now() - started).toFixed(0)} ms`);
          return {
            model: data.model,
            prompt: data.prompt,
            vector: Float32Array.from(data.embedding),
          };
        },
      });
      const totals = subtreeSplats(doc.instances);
      const store = useInstances.getState();
      store.setTable(scan, {
        ...doc,
        embeddingSource: { instancesUrl: `file://${dir}/instances.json`, ref: doc.embedding },
      });
      const show = (r: SearchResult): string => {
        const i = doc.byId.get(r.id);
        return `#${String(r.id)} L${String(i?.level)} ${r.score.toFixed(3)} ${r.via} "${r.label}" (${String(totals.get(r.id))} splats)`;
      };
      for (const query of QUERIES[scan] ?? []) {
        const tagsOnly = searchInstances(doc.instances, query).slice(0, 5);
        useInstances.getState().setQuery(scan, query);
        const started = performance.now();
        while (useInstances.getState().assets[scan]?.meaning === "loading") {
          await new Promise((resolve) => setTimeout(resolve, 5));
          if (performance.now() - started > 20_000) throw new Error("meaning never arrived");
        }
        const entry = useInstances.getState().assets[scan];
        expect(entry?.meaning).toBe("ready");
        const top = (entry?.results ?? []).slice(0, 5);
        console.info(
          `${scan} "${query}" (${(performance.now() - started).toFixed(0)} ms)\n  tags only:  ${
            tagsOnly.map(show).join("\n              ") || "(nothing)"
          }\n  + meaning:  ${top.map(show).join("\n              ")}`,
        );
        expect(top.length).toBeGreaterThan(0);
      }
    }, 60_000);
  }
});
