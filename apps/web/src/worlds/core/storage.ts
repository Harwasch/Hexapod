import {
  DEFAULT_SETTINGS,
  newId,
  type AppSettings,
  type LibraryRecords,
  type LibraryStore,
  type MediaAsset,
} from "./types";

const DB_NAME = "hexapod-worlds-local";
const STORES: LibraryStore[] = [
  "projects",
  "scenes",
  "characters",
  "replays",
  "assets",
  "benchmarks",
  "reconstructions",
];
const MAX_ARCHIVE_BYTES = 150 * 1024 * 1024;
const SETTINGS_KEY = "hexapod.worlds.settings.v1";
const TOKEN_KEY = "hexapod.worlds.session-token";
let database: Promise<IDBDatabase> | undefined;
export class LibraryError extends Error {
  constructor(
    message: string,
    public readonly code: "unavailable" | "quota" | "invalid" | "storage",
  ) {
    super(message);
    this.name = "LibraryError";
  }
}
function storageError(error: unknown): LibraryError {
  if (error instanceof LibraryError) return error;
  if (error instanceof DOMException && error.name === "QuotaExceededError")
    return new LibraryError(
      "Your browser storage is full. Export your library, then remove unused recordings or media.",
      "quota",
    );
  return new LibraryError(
    "Local library storage failed. Your browser may be blocking storage or running in private mode.",
    "storage",
  );
}
function openDatabase(): Promise<IDBDatabase> {
  if (database) return database;
  database = new Promise((resolve, reject) => {
    if (!globalThis.indexedDB) {
      reject(
        new LibraryError(
          "This browser does not support the local library. Use a browser with IndexedDB enabled.",
          "unavailable",
        ),
      );
      return;
    }
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      for (const store of [...STORES, "blobs"])
        if (!request.result.objectStoreNames.contains(store))
          request.result.createObjectStore(store, { keyPath: "id" });
    };
    request.onsuccess = () => {
      request.result.onversionchange = () => {
        request.result.close();
        database = undefined;
      };
      resolve(request.result);
    };
    request.onerror = () => reject(storageError(request.error));
    request.onblocked = () =>
      reject(
        new LibraryError(
          "Another Worlds tab is holding an older library open. Close it and try again.",
          "unavailable",
        ),
      );
  });
  database.catch(() => {
    database = undefined;
  });
  return database;
}
function completed(transaction: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(storageError(transaction.error));
    transaction.onabort = () => reject(storageError(transaction.error));
  });
}
function result<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(storageError(request.error));
  });
}
const listeners = new Set<() => void>();
function changed(): void {
  listeners.forEach((listener) => listener());
}
function invalid(message: string): never {
  throw new LibraryError(message, "invalid");
}
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
function stringField(record: Record<string, unknown>, key: string, max = 100_000): void {
  if (typeof record[key] !== "string" || record[key].length > max)
    invalid(`Invalid ${key} in imported library.`);
}
function numberField(record: Record<string, unknown>, key: string): void {
  if (typeof record[key] !== "number" || !Number.isFinite(record[key]) || record[key] < 0)
    invalid(`Invalid ${key} in imported library.`);
}
function stringsField(record: Record<string, unknown>, key: string): void {
  if (
    !Array.isArray(record[key]) ||
    record[key].length > 10_000 ||
    !(record[key] as unknown[]).every((value) => typeof value === "string")
  )
    invalid(`Invalid ${key} in imported library.`);
}
function eventsField(record: Record<string, unknown>): void {
  if (!Array.isArray(record.events) || record.events.length > 100_000)
    invalid("Invalid event trajectory.");
  for (const value of record.events) {
    if (!isRecord(value)) invalid("Invalid control event.");
    stringField(value, "id", 200);
    numberField(value, "timestampMs");
    if (!["native", "prompt", "semantic", "pause", "resume"].includes(String(value.type)))
      invalid("Unsupported control event.");
    for (const key of ["action", "prompt"]) if (value[key] !== undefined) stringField(value, key);
    if (
      value.values !== undefined &&
      (!isRecord(value.values) ||
        Object.values(value.values).some(
          (item) =>
            !["string", "number", "boolean"].includes(typeof item) ||
            (typeof item === "number" && !Number.isFinite(item)),
        ))
    )
      invalid("Invalid action vector.");
  }
}
export function validateRecord<K extends LibraryStore>(
  store: K,
  value: unknown,
): asserts value is LibraryRecords[K] {
  if (!isRecord(value)) invalid(`Invalid ${store} record.`);
  stringField(value, "id", 200);
  if (!value.id) invalid("Records need an ID.");
  stringField(value, "name", 500);
  numberField(value, "createdAt");
  if (store === "projects") {
    for (const key of ["prompt", "modelId", "providerId"]) stringField(value, key);
    if (
      !["runpod", "local", "modal", "lambda", "coreweave", "aws", "gcp", "azure"].includes(
        String(value.providerId),
      )
    )
      invalid("Unknown compute provider.");
    stringsField(value, "assetIds");
    stringsField(value, "characterIds");
    numberField(value, "updatedAt");
    if (value.game !== undefined) {
      if (!isRecord(value.game)) invalid("Invalid game concept.");
      stringField(value.game, "objective", 1000);
      if (!Array.isArray(value.game.events) || value.game.events.length > 12)
        invalid("Invalid game event schedule.");
      for (const event of value.game.events) {
        if (!isRecord(event)) invalid("Invalid game event.");
        numberField(event, "atSeconds");
        if (Number(event.atSeconds) > 3600) invalid("Game event exceeds one hour.");
        stringField(event, "prompt", 1000);
      }
    }
    if (value.referencePreparation !== undefined) {
      if (
        !isRecord(value.referencePreparation) ||
        !["last-video-frame", "image-synthesis"].includes(String(value.referencePreparation.method))
      )
        invalid("Invalid reference preparation.");
      stringsField(value.referencePreparation, "sourceAssetIds");
    }
    if (
      !isRecord(value.settings) ||
      !["quality", "balanced", "low-latency"].includes(String(value.settings.performance))
    )
      invalid("Invalid world settings.");
    if (
      value.settings.seed !== undefined &&
      (!Number.isSafeInteger(value.settings.seed) || Number(value.settings.seed) < 0)
    )
      invalid("Seed must be a nonnegative safe integer.");
  } else if (store === "characters") {
    stringField(value, "description");
    stringsField(value, "assetIds");
    numberField(value, "updatedAt");
  } else if (store === "assets") {
    stringField(value, "mimeType", 200);
    numberField(value, "size");
    if (!["image", "video", "audio", "snapshot", "reconstruction"].includes(String(value.kind)))
      invalid("Unknown media type.");
  } else {
    stringField(value, "projectId", 200);
    if (store === "scenes") {
      stringField(value, "modelId", 200);
      stringField(value, "prompt");
      stringsField(value, "assetIds");
      eventsField(value);
      if (!["exact", "approximate", "visual"].includes(String(value.resumeKind)))
        invalid("Unknown scene resume classification.");
      if (value.snapshot !== undefined && !isRecord(value.snapshot))
        invalid("Invalid scene snapshot.");
    } else if (store === "replays") {
      stringField(value, "assetId", 200);
      stringField(value, "modelId", 200);
      numberField(value, "durationMs");
      eventsField(value);
    } else if (store === "benchmarks") {
      for (const key of ["modelId", "providerId"]) stringField(value, key, 200);
      numberField(value, "durationMs");
      numberField(value, "frameCount");
      eventsField(value);
      for (const key of ["measuredFPS", "latencyMs", "estimatedCostUSD"])
        if (value[key] !== undefined) numberField(value, key);
      if (
        value.ratings !== undefined &&
        (!isRecord(value.ratings) ||
          Object.values(value.ratings).some(
            (rating) =>
              typeof rating !== "number" || !Number.isFinite(rating) || rating < 1 || rating > 5,
          ))
      )
        invalid("Benchmark ratings must be between 1 and 5.");
    } else if (store === "reconstructions") {
      stringsField(value, "sourceAssetIds");
      if (
        !["queued", "running", "completed", "failed"].includes(String(value.status)) ||
        !["ply", "spz", "glb"].includes(String(value.format))
      )
        invalid("Invalid reconstruction record.");
    }
  }
  for (const key of ["thumbnailAssetId", "parentSceneId", "jobId", "error", "notes"])
    if (value[key] !== undefined) stringField(value, key);
}
export interface LibraryArchive {
  format: "hexapod-worlds";
  version: 1;
  exportedAt: number;
  records: { [K in LibraryStore]: LibraryRecords[K][] };
  blobs: { id: string; mimeType: string; base64: string }[];
}
// Keep binary conversion bounded: a single long recording must not allocate a
// second full-sized binary string alongside its base64 representation.
const ENCODE_CHUNK_BYTES = 3 * 16_384;
const DECODE_CHUNK_CHARACTERS = 4 * 16_384;
async function encodedBlobParts(blob: Blob): Promise<Blob[]> {
  const parts: Blob[] = [];
  for (let offset = 0; offset < blob.size; offset += ENCODE_CHUNK_BYTES) {
    const bytes = new Uint8Array(
      await blob.slice(offset, offset + ENCODE_CHUNK_BYTES).arrayBuffer(),
    );
    parts.push(new Blob([btoa(String.fromCharCode(...bytes))]));
  }
  return parts;
}
function decodedBlob(encoded: string, mimeType: string): Blob {
  const parts: Blob[] = [];
  for (let offset = 0; offset < encoded.length; offset += DECODE_CHUNK_CHARACTERS) {
    const binary = atob(encoded.slice(offset, offset + DECODE_CHUNK_CHARACTERS));
    const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
    parts.push(new Blob([bytes]));
  }
  return new Blob(parts, { type: mimeType });
}
export function parseLibraryArchive(text: string): LibraryArchive {
  if (text.length > MAX_ARCHIVE_BYTES * 1.4)
    invalid("Library archive exceeds the 150 MB import limit. Export large videos separately.");
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return invalid("This file is not valid JSON.");
  }
  if (
    !isRecord(parsed) ||
    parsed.format !== "hexapod-worlds" ||
    parsed.version !== 1 ||
    !isRecord(parsed.records) ||
    !Array.isArray(parsed.blobs)
  )
    invalid("Unsupported library archive. Expected a Worlds v1 export.");
  let total = 0;
  for (const store of STORES) {
    const records: unknown = parsed.records[store];
    if (!Array.isArray(records) || records.length > 10_000) invalid(`Invalid ${store} collection.`);
    const ids = new Set<string>();
    for (const record of records) {
      validateRecord(store, record);
      if (ids.has(record.id)) invalid("Duplicate record IDs in archive.");
      ids.add(record.id);
    }
  }
  const archive = parsed as unknown as LibraryArchive;
  const assetIds = new Set(archive.records.assets.map((asset) => asset.id));
  const blobIds = new Set<string>();
  for (const blob of parsed.blobs) {
    if (!isRecord(blob)) invalid("Invalid media blob.");
    stringField(blob, "id", 200);
    stringField(blob, "mimeType", 200);
    stringField(blob, "base64", MAX_ARCHIVE_BYTES * 1.4);
    if (!assetIds.has(String(blob.id)) || blobIds.has(String(blob.id)))
      invalid("Media blob references an unknown or duplicate asset.");
    const encoded = String(blob.base64);
    if (
      encoded.length % 4 !== 0 ||
      /[^A-Za-z0-9+/=]/.test(encoded) ||
      encoded.slice(0, -2).includes("=") ||
      (!encoded.endsWith("=") && encoded.includes("=")) ||
      (encoded.endsWith("==") && encoded.length < 4)
    )
      invalid("Corrupted media encoding.");
    total += String(blob.base64).length * 0.75;
    if (total > MAX_ARCHIVE_BYTES) invalid("Media in archive exceeds the 150 MB import limit.");
    blobIds.add(String(blob.id));
  }
  for (const asset of archive.records.assets)
    if (!blobIds.has(asset.id)) invalid("Archive is missing media data.");
  const requireAssets = (ids: string[]) => {
    if (ids.some((id) => !assetIds.has(id))) invalid("Archive references missing media assets.");
  };
  const characterIds = new Set(archive.records.characters.map((character) => character.id));
  for (const project of archive.records.projects) {
    requireAssets(project.assetIds);
    if (project.thumbnailAssetId) requireAssets([project.thumbnailAssetId]);
    if (project.characterIds.some((id) => !characterIds.has(id)))
      invalid("Archive references missing characters.");
  }
  for (const character of archive.records.characters) requireAssets(character.assetIds);
  for (const scene of archive.records.scenes) {
    requireAssets(scene.assetIds);
    if (scene.thumbnailAssetId) requireAssets([scene.thumbnailAssetId]);
  }
  for (const replay of archive.records.replays) requireAssets([replay.assetId]);
  for (const reconstruction of archive.records.reconstructions) {
    requireAssets(reconstruction.sourceAssetIds);
    if (reconstruction.assetId) requireAssets([reconstruction.assetId]);
  }
  return archive;
}
export const worldStore = {
  subscribe(listener: () => void): () => void {
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  },
  async list<K extends LibraryStore>(store: K): Promise<LibraryRecords[K][]> {
    const db = await openDatabase();
    return (
      await result<LibraryRecords[K][]>(
        db.transaction(store).objectStore(store).getAll() as IDBRequest<LibraryRecords[K][]>,
      )
    ).sort((a, b) => b.createdAt - a.createdAt);
  },
  async get<K extends LibraryStore>(store: K, id: string): Promise<LibraryRecords[K] | undefined> {
    const db = await openDatabase();
    return result(
      db.transaction(store).objectStore(store).get(id) as IDBRequest<LibraryRecords[K] | undefined>,
    );
  },
  async put<K extends LibraryStore>(store: K, record: LibraryRecords[K]): Promise<void> {
    validateRecord(store, record);
    const db = await openDatabase();
    const tx = db.transaction(store, "readwrite");
    const done = completed(tx);
    tx.objectStore(store).put(record);
    await done;
    changed();
  },
  async remove(store: LibraryStore, id: string): Promise<void> {
    const db = await openDatabase();
    const tx = db.transaction([...STORES, "blobs"], "readwrite");
    const done = completed(tx);
    const collections = await Promise.all(
      STORES.map((key) => result(tx.objectStore(key).getAll())),
    );
    if (store === "assets") {
      const referenced = collections.some(
        (records, index) =>
          STORES[index] !== "assets" &&
          records.some(
            (record: Record<string, unknown>) =>
              record.thumbnailAssetId === id ||
              record.assetId === id ||
              (Array.isArray(record.assetIds) && record.assetIds.includes(id)) ||
              (Array.isArray(record.sourceAssetIds) && record.sourceAssetIds.includes(id)),
          ),
      );
      if (referenced) {
        tx.abort();
        await done.catch(() => undefined);
        throw new LibraryError(
          "This media is still used by a world, character, scene, replay, or reconstruction. Remove those references before deleting it.",
          "invalid",
        );
      }
      tx.objectStore("blobs").delete(id);
    }
    if (store === "characters" || store === "scenes") {
      for (const project of collections[0] as LibraryRecords["projects"][]) {
        if (store === "characters" && project.characterIds.includes(id))
          tx.objectStore("projects").put({
            ...project,
            characterIds: project.characterIds.filter((characterId) => characterId !== id),
            updatedAt: Date.now(),
          });
        if (store === "scenes" && project.parentSceneId === id)
          tx.objectStore("projects").put({
            ...project,
            parentSceneId: undefined,
            updatedAt: Date.now(),
          });
      }
    }
    if (store === "projects") {
      for (const childStore of ["scenes", "replays", "benchmarks", "reconstructions"] as const) {
        for (const child of collections[STORES.indexOf(childStore)] as {
          id: string;
          projectId: string;
        }[])
          if (child.projectId === id) tx.objectStore(childStore).delete(child.id);
      }
      // Assets remain available for other projects and in the portable library.
    }
    tx.objectStore(store).delete(id);
    await done;
    changed();
  },
  async saveAsset(
    blob: Blob,
    name = "Untitled media",
    kind?: MediaAsset["kind"],
  ): Promise<MediaAsset> {
    const asset: MediaAsset = {
      id: newId(),
      name,
      kind:
        kind ??
        (blob.type.startsWith("video/")
          ? "video"
          : blob.type.startsWith("audio/")
            ? "audio"
            : "image"),
      mimeType: blob.type || "application/octet-stream",
      size: blob.size,
      createdAt: Date.now(),
    };
    const db = await openDatabase();
    const tx = db.transaction(["assets", "blobs"], "readwrite");
    const done = completed(tx);
    tx.objectStore("assets").put(asset);
    tx.objectStore("blobs").put({
      id: asset.id,
      blob: blob.type ? blob : new Blob([blob], { type: asset.mimeType }),
    });
    await done;
    changed();
    return asset;
  },
  async getBlob(id: string): Promise<Blob | undefined> {
    const db = await openDatabase();
    const row = await result<{ id: string; blob: Blob } | undefined>(
      db.transaction("blobs").objectStore("blobs").get(id) as IDBRequest<
        { id: string; blob: Blob } | undefined
      >,
    );
    return row?.blob;
  },
  async exportLibrary(): Promise<Blob> {
    const db = await openDatabase();
    const tx = db.transaction([...STORES, "blobs"]);
    const records = {} as LibraryArchive["records"];
    const allRecords = await Promise.all(
      STORES.map((store) => result(tx.objectStore(store).getAll())).concat([
        result(tx.objectStore("blobs").getAll()),
      ]),
    );
    STORES.forEach((store, index) => {
      (records[store] as unknown[]) = allRecords[index] ?? [];
    });
    const blobs = (allRecords[STORES.length] ?? []) as { id: string; blob: Blob }[];
    if (blobs.reduce((sum, item) => sum + item.blob.size, 0) > MAX_ARCHIVE_BYTES)
      throw new LibraryError(
        "This library exceeds the 150 MB archive limit. Download large replay videos separately and remove them before exporting.",
        "invalid",
      );
    let envelope: string;
    try {
      envelope = JSON.stringify({
        format: "hexapod-worlds",
        version: 1,
        exportedAt: Date.now(),
        records,
      });
    } catch {
      throw new LibraryError(
        "The library contains a snapshot that cannot be exported as JSON.",
        "invalid",
      );
    }
    const parts: BlobPart[] = [envelope.slice(0, -1), ',"blobs":['];
    for (const [index, row] of blobs.entries()) {
      parts.push(
        index ? "," : "",
        JSON.stringify({ id: row.id, mimeType: row.blob.type }).slice(0, -1),
        ',"base64":"',
      );
      parts.push(...(await encodedBlobParts(row.blob)), '"}');
    }
    parts.push("]}");
    const output = new Blob(parts, { type: "application/json" });
    if (output.size > MAX_ARCHIVE_BYTES * 1.4)
      throw new LibraryError(
        "Library metadata and media exceed the portable archive size limit. Download large replays separately before exporting.",
        "invalid",
      );
    return output;
  },
  /** Validates the entire archive before a single atomic merge. Matching IDs are replaced. */
  async importLibrary(input: string | Blob): Promise<number> {
    if (input instanceof Blob && input.size > MAX_ARCHIVE_BYTES * 1.4)
      invalid("Library archive exceeds the import limit.");
    const archive = parseLibraryArchive(typeof input === "string" ? input : await input.text());
    const metadataById = new Map(archive.records.assets.map((asset) => [asset.id, asset]));
    // Validate byte lengths before allocating decoded media or opening a transaction.
    for (const item of archive.blobs) {
      const metadata = metadataById.get(item.id);
      const padding = item.base64.endsWith("==") ? 2 : item.base64.endsWith("=") ? 1 : 0;
      const size = (item.base64.length * 3) / 4 - padding;
      if (metadata?.size !== size || metadata.mimeType !== item.mimeType)
        invalid("Media size or type does not match its metadata.");
    }
    const decoded = archive.blobs.map((item) => ({
      id: item.id,
      blob: decodedBlob(item.base64, item.mimeType),
    }));
    const db = await openDatabase();
    const tx = db.transaction([...STORES, "blobs"], "readwrite");
    const done = completed(tx);
    let count = 0;
    for (const store of STORES)
      for (const record of archive.records[store]) {
        tx.objectStore(store).put(record);
        count += 1;
      }
    for (const row of decoded) tx.objectStore("blobs").put(row);
    await done;
    changed();
    return count;
  },
  async estimateStorage(): Promise<StorageEstimate | undefined> {
    return navigator.storage?.estimate?.();
  },
  async requestPersistentStorage(): Promise<boolean> {
    return navigator.storage?.persist?.() ?? false;
  },
};
export function getSettings(): AppSettings {
  try {
    const raw: unknown = JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? "{}");
    if (!isRecord(raw)) return { ...DEFAULT_SETTINGS };
    return {
      apiBaseUrl: typeof raw.apiBaseUrl === "string" ? raw.apiBaseUrl : "",
      defaultProvider: [
        "runpod",
        "local",
        "modal",
        "lambda",
        "coreweave",
        "aws",
        "gcp",
        "azure",
      ].includes(String(raw.defaultProvider))
        ? (raw.defaultProvider as AppSettings["defaultProvider"])
        : "runpod",
      idleTimeoutMinutes:
        typeof raw.idleTimeoutMinutes === "number" &&
        raw.idleTimeoutMinutes >= 1 &&
        raw.idleTimeoutMinutes <= 240
          ? raw.idleTimeoutMinutes
          : 10,
      retainWorker: raw.retainWorker === true,
    };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}
export function saveSettings(patch: Partial<AppSettings>): AppSettings {
  const settings = { ...getSettings(), ...patch };
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch (error) {
    throw storageError(error);
  }
  return settings;
}
/** Session manager access token is never part of library export or localStorage. */
export function getApiToken(): string {
  try {
    return sessionStorage.getItem(TOKEN_KEY) ?? "";
  } catch {
    return "";
  }
}
export function setApiToken(token: string): void {
  try {
    if (token) sessionStorage.setItem(TOKEN_KEY, token);
    else sessionStorage.removeItem(TOKEN_KEY);
  } catch (error) {
    throw storageError(error);
  }
}
export function downloadBlob(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
}
