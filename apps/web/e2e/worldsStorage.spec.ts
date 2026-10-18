import { expect, test } from "@playwright/test";
import type * as WorldsStorage from "../src/worlds/core/storage";

/** Real IndexedDB and Blob regression, independent of GPU services and the Earth UI. */
test("Worlds library preserves blobs, rejects corrupt imports, and maintains references on delete", async ({
  page,
}) => {
  await page.route("**/worlds-storage-test", (route) =>
    route.fulfill({
      contentType: "text/html",
      body: "<!doctype html><title>Worlds storage test</title>",
    }),
  );
  await page.goto("/worlds-storage-test");
  const result = await page.evaluate(async () => {
    const modulePath = "/src/worlds/core/storage.ts";
    const { worldStore, setApiToken, parseLibraryArchive } = (await import(
      modulePath
    )) as typeof WorldsStorage;
    setApiToken("private-test-session-token");
    const chunkedBytes = Uint8Array.from({ length: 150_001 }, (_, index) => index % 256);
    const chunkedMedia = await worldStore.saveAsset(
      new Blob([chunkedBytes], { type: "video/webm" }),
      "multi-chunk.webm",
      "video",
    );
    const media = await worldStore.saveAsset(
      new Blob(["original-media-bytes"]),
      "reference.bin",
      "snapshot",
    );
    await worldStore.put("characters", {
      id: "character",
      name: "Explorer",
      description: "A yellow jacket",
      assetIds: [media.id],
      createdAt: 1,
      updatedAt: 1,
    });
    await worldStore.put("projects", {
      id: "world",
      name: "Forest",
      prompt: "A moonlit forest",
      modelId: "astronex-world",
      providerId: "runpod",
      createdAt: 1,
      updatedAt: 1,
      assetIds: [media.id],
      characterIds: ["character"],
      settings: { performance: "balanced", seed: 0 },
    });
    await worldStore.put("scenes", {
      id: "scene",
      name: "First clearing",
      projectId: "world",
      modelId: "astronex-world",
      prompt: "A moonlit forest",
      createdAt: 2,
      assetIds: [media.id],
      thumbnailAssetId: media.id,
      events: [],
      resumeKind: "visual",
    });
    await worldStore.put("replays", {
      id: "replay",
      name: "A walk",
      projectId: "world",
      modelId: "astronex-world",
      createdAt: 3,
      assetId: media.id,
      durationMs: 1200,
      events: [],
    });
    await worldStore.put("benchmarks", {
      id: "benchmark",
      name: "Measured sample",
      projectId: "world",
      modelId: "astronex-world",
      providerId: "runpod",
      createdAt: 4,
      durationMs: 1200,
      frameCount: 0,
      events: [],
    });
    await worldStore.put("reconstructions", {
      id: "reconstruction",
      name: "Clearing geometry",
      projectId: "world",
      createdAt: 5,
      status: "queued",
      format: "ply",
      sourceAssetIds: [media.id],
    });

    const originalArchive = await (await worldStore.exportLibrary()).text();
    let protectedAsset = false;
    try {
      await worldStore.remove("assets", media.id);
    } catch (error) {
      protectedAsset = error instanceof Error && error.message.includes("still used");
    }
    const preservedAfterDeleteRejection = await (await worldStore.getBlob(media.id))?.text();
    await worldStore.remove("characters", "character");
    const detachedCharacterIds = (await worldStore.get("projects", "world"))?.characterIds;
    const afterCharacterDelete = parseLibraryArchive(
      await (await worldStore.exportLibrary()).text(),
    );
    const deleteRoundtripCount = await worldStore.importLibrary(
      JSON.stringify(afterCharacterDelete),
    );

    await worldStore.remove("projects", "world");
    const childCounts = await Promise.all([
      worldStore.list("scenes"),
      worldStore.list("replays"),
      worldStore.list("benchmarks"),
      worldStore.list("reconstructions"),
    ]).then((lists) => lists.map((list) => list.length));
    await worldStore.remove("assets", media.id);
    const absentBlob = (await worldStore.getBlob(media.id)) === undefined;
    await worldStore.remove("assets", chunkedMedia.id);
    await worldStore.importLibrary(originalArchive);
    const chunkedRestored = await worldStore.getBlob(chunkedMedia.id);
    const chunkedRoundtrip =
      !!chunkedRestored &&
      new Uint8Array(await chunkedRestored.arrayBuffer()).every(
        (byte, index) => byte === index % 256,
      ) &&
      chunkedRestored.size === 150_001;
    const restoredBlob = await worldStore.getBlob(media.id);
    const restoredBytes = await restoredBlob?.text();
    const restoredMime = restoredBlob?.type;

    const corrupted = JSON.parse(originalArchive) as {
      records: { assets: { size: number }[]; projects: { prompt: string }[] };
    };
    const firstAsset = corrupted.records.assets[0];
    const firstProject = corrupted.records.projects[0];
    if (!firstAsset || !firstProject) throw new Error("Missing test records");
    firstAsset.size = 999;
    firstProject.prompt = "Must never be committed";
    let rejectedCorruption = false;
    try {
      await worldStore.importLibrary(JSON.stringify(corrupted));
    } catch (error) {
      rejectedCorruption = error instanceof Error && error.message.includes("does not match");
    }
    const preservedPrompt = (await worldStore.get("projects", "world"))?.prompt;
    return {
      chunkedRoundtrip,
      protectedAsset,
      preservedAfterDeleteRejection,
      detachedCharacterIds,
      deleteRoundtripCount,
      childCounts,
      absentBlob,
      restoredBytes,
      restoredMime,
      rejectedCorruption,
      preservedPrompt,
      tokenLeaked: originalArchive.includes("private-test-session-token"),
    };
  });
  expect(result).toMatchObject({
    chunkedRoundtrip: true,
    protectedAsset: true,
    preservedAfterDeleteRejection: "original-media-bytes",
    detachedCharacterIds: [],
    childCounts: [0, 0, 0, 0],
    absentBlob: true,
    restoredBytes: "original-media-bytes",
    restoredMime: "application/octet-stream",
    rejectedCorruption: true,
    preservedPrompt: "A moonlit forest",
    tokenLeaked: false,
  });
  expect(result.deleteRoundtripCount).toBeGreaterThan(0);
});
