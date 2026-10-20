/**
 * Search by meaning on real segmented scans, through the real API encoder. Skipped unless
 * both are given:
 *
 *   INSTANCES_MEANING_DIRS=/path/spool,/path/pumpkin,/path/camp   (instances.json + .emb)
 *   INSTANCES_MEANING_API=http://127.0.0.1:8000                   (TEXT_ENCODER_DIR set)
 *   INSTANCES_MEANING_QUERIES='spool=spool|cable spool;camp=cabin|flagpole'  (optional)
 *
 * It drives the store (`state/instances.ts`) as the panel does and prints each query's top
 * five objects, by tags alone and by tags and meaning.
 */
import { readFileSync } from "node:fs";
import { basename, join } from "node:path";

import { afterAll, describe, expect, it } from "vitest";

import type { QueryEmbedding } from "@/api/textEmbedding";
import { parseInstances } from "@/lib/instances";
import { configureMeaning, matchObjects, useInstances } from "@/state/instances";

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
        loadEmbeddings: () =>
          Promise.resolve(new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.length / 2)),
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
      const { embedding } = doc;
      useInstances.getState().setTable(scan, {
        ...doc,
        embeddingSource: {
          url: `file://${dir}/${embedding.file}`,
          rows: bytes.length / (2 * embedding.dim),
          dim: embedding.dim,
          model: embedding.model,
        },
      });
      for (const query of QUERIES[scan] ?? []) {
        const entryNow = useInstances.getState().assets[scan];
        if (!entryNow) throw new Error("no table");
        const tagsOnly = matchObjects(entryNow, query).slice(0, 5);
        useInstances.getState().setQuery(scan, query);
        const started = performance.now();
        while (useInstances.getState().assets[scan]?.meaning === "loading") {
          await new Promise((resolve) => setTimeout(resolve, 5));
          if (performance.now() - started > 20_000) throw new Error("meaning never arrived");
        }
        const entry = useInstances.getState().assets[scan];
        expect(entry?.meaning).toBe("ready");
        const top = (entry?.matches ?? []).slice(0, 5);
        const show = (id: number): string => {
          const object = entry?.index.objects.get(id);
          const via = entry?.byMeaning.has(id) ? "meaning" : "tags";
          return `#${String(id)} ${via} "${object?.name ?? "?"}" (${String(object?.splats)} splats)`;
        };
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
