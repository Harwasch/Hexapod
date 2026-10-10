export const VIEWER_MAX_BYTES = 128 * 1024 * 1024;
export const VIEWER_MAX_POINTS = 2_000_000;
const TOO_LARGE =
  "This file is too large for the embedded viewer. Download it for a desktop 3D tool.";

/** Enforce actual streamed bytes; Content-Length may be missing or inaccurate. */
export async function boundedArtifactBlob(
  response: Response,
  limit = VIEWER_MAX_BYTES,
): Promise<Blob> {
  if (Number(response.headers.get("content-length")) > limit) {
    await response.body?.cancel();
    throw new Error(TOO_LARGE);
  }
  if (!response.body) throw new Error("The download returned no geometry.");
  const reader = response.body.getReader();
  const chunks: Uint8Array<ArrayBuffer>[] = [];
  let bytes = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes += value.byteLength;
      if (bytes > limit) {
        await reader.cancel();
        throw new Error(TOO_LARGE);
      }
      chunks.push(new Uint8Array(value));
    }
  } finally {
    reader.releaseLock();
  }
  if (!bytes) throw new Error("The download returned no geometry.");
  return new Blob(chunks, {
    type: response.headers.get("content-type") ?? "application/octet-stream",
  });
}

/** Check untrusted PLY allocation declarations before handing them to a renderer. */
export function validatePlyHeader(bytes: ArrayBuffer): { gaussian: boolean } {
  const header = new TextDecoder().decode(bytes.slice(0, 64 * 1024));
  const end = header.indexOf("end_header");
  if (!header.startsWith("ply\n") && !header.startsWith("ply\r\n"))
    throw new Error("This file is not a PLY.");
  if (end < 0) throw new Error("The PLY header is missing or too large.");
  const properties = header.slice(0, end);
  const match = /^element vertex (\d+)\s*$/m.exec(properties);
  const count = Number(match?.[1]);
  if (!Number.isSafeInteger(count) || count < 1 || count > VIEWER_MAX_POINTS)
    throw new Error("Choose geometry containing between 1 and 2 million points for this viewer.");
  if (
    /^property list /m.test(properties) ||
    [...properties.matchAll(/^element (\w+) (\d+)\s*$/gm)].some(
      (element) => element[1] !== "vertex" && Number(element[2]) > 0,
    )
  )
    throw new Error(
      "Open a point-cloud or Gaussian PLY. Mesh faces and list properties are not supported in this viewer.",
    );
  return { gaussian: /^property float f_dc_0\s*$/m.test(properties) };
}

/** SPZ is compressed: constrain its expanded data and declared allocations as well. */
export async function validateSpz(bytes: ArrayBuffer): Promise<void> {
  if (typeof DecompressionStream === "undefined")
    throw new Error("This browser cannot open SPZ. Download a PLY export instead.");
  const reader = new Blob([bytes])
    .stream()
    .pipeThrough(new DecompressionStream("gzip"))
    .getReader();
  const header = new Uint8Array(16);
  let copied = 0;
  let expanded = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      expanded += value.byteLength;
      if (expanded > 256 * 1024 * 1024)
        throw new Error("The expanded SPZ exceeds this viewer's memory limit.");
      if (copied < 16) {
        const part = value.subarray(0, 16 - copied);
        header.set(part, copied);
        copied += part.byteLength;
        if (copied === 16) {
          const view = new DataView(header.buffer);
          if (view.getUint32(0, true) !== 0x5053474e || ![2, 3].includes(view.getUint32(4, true)))
            throw new Error("This viewer accepts gzip-compressed SPZ version 2 or 3.");
          const count = view.getUint32(8, true);
          if (!count || count > VIEWER_MAX_POINTS)
            throw new Error(
              "Choose an SPZ containing between 1 and 2 million points for this viewer.",
            );
        }
      }
    }
    if (copied < 16) throw new Error("The SPZ header is incomplete.");
  } finally {
    await reader.cancel();
    reader.releaseLock();
  }
}
