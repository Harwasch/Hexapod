/** Portable, uncompressed POSIX tar: no upload and no archive dependency. */
export interface ReconstructionSource {
  id: string;
  blob: Blob;
  name: string;
  timestampMs?: number;
  /** Only supplied if the model actually emitted it. */
  cameraPose?: number[];
  assetId?: string;
}

export interface ReconstructionManifest {
  schema: "worlds-reconstruction/v1";
  createdAt: string;
  modelId?: string;
  source: "generated-world";
  geometry: "unverified";
  coordinateSystem: "unknown";
  frames: {
    path: string;
    mimeType: string;
    bytes: number;
    timestampMs?: number;
    cameraPose?: number[];
  }[];
  notes: string[];
}

const encoder = new TextEncoder();
const extensions: Record<string, string> = {
  "image/jpeg": "jpg",
  "image/png": "png",
  "image/webp": "webp",
  "video/webm": "webm",
  "video/mp4": "mp4",
  "video/quicktime": "mov",
};

/** Random/local filenames, filesystem paths and credentials never enter exported media paths. */
export function manifestFor(
  sources: ReconstructionSource[],
  modelId?: string,
): ReconstructionManifest {
  return {
    schema: "worlds-reconstruction/v1",
    createdAt: new Date().toISOString(),
    ...(modelId ? { modelId } : {}),
    source: "generated-world",
    geometry: "unverified",
    coordinateSystem: "unknown",
    frames: sources.map((source, index) => ({
      path: `sources/${String(index + 1).padStart(5, "0")}.${extensions[source.blob.type.split(";")[0] ?? ""] ?? "bin"}`,
      mimeType: source.blob.type,
      bytes: source.blob.size,
      ...(source.timestampMs !== undefined ? { timestampMs: source.timestampMs } : {}),
      ...(source.cameraPose ? { cameraPose: source.cameraPose } : {}),
    })),
    notes: [
      "Source media only. No reconstructed geometry is included.",
      "Generated imagery may drift, deform, or fail multi-view reconstruction.",
      "Camera poses and depth must be estimated unless explicitly supplied by the model.",
      "Prefer a static scene, overlapping views and real camera translation; avoid cuts and moving subjects.",
    ],
  };
}

function header(path: string, bytes: number): Uint8Array<ArrayBuffer> {
  const block = new Uint8Array(512);
  const write = (text: string, offset: number): void => {
    block.set(encoder.encode(text), offset);
  };
  const octal = (value: number, length: number): string =>
    value.toString(8).padStart(length - 1, "0") + "\0";
  write(path, 0);
  write(octal(0o644, 8), 100);
  write(octal(0, 8), 108);
  write(octal(0, 8), 116);
  write(octal(bytes, 12), 124);
  write(octal(0, 12), 136);
  write("        ", 148);
  write("0", 156);
  write("ustar\0", 257);
  write("00", 263);
  const sum = block.reduce((total, value) => total + value, 0);
  write(sum.toString(8).padStart(6, "0") + "\0 ", 148);
  return block;
}

export function createReconstructionBundle(
  sources: ReconstructionSource[],
  modelId?: string,
): Blob {
  if (!sources.length) throw new Error("Select a replay or at least one frame first.");
  const manifest = manifestFor(sources, modelId);
  const files = [
    {
      path: "manifest.json",
      blob: new Blob([JSON.stringify(manifest, null, 2)], { type: "application/json" }),
    },
    ...sources.map((source, index) => ({
      path: manifest.frames[index]?.path ?? "",
      blob: source.blob,
    })),
  ];
  const parts: BlobPart[] = [];
  for (const file of files) {
    parts.push(header(file.path, file.blob.size), file.blob);
    const remainder = file.blob.size % 512;
    if (remainder) parts.push(new Uint8Array(512 - remainder));
  }
  parts.push(new Uint8Array(1024));
  return new Blob(parts, { type: "application/x-tar" });
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
}
