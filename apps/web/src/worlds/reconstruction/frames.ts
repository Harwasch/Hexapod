import type { ReconstructionSource } from "./bundle";

function waitFor(video: HTMLVideoElement, event: string, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const cleanup = (): void => {
      clearTimeout(timeout);
      video.removeEventListener(event, ready);
      video.removeEventListener("error", failed);
      signal?.removeEventListener("abort", aborted);
    };
    const ready = (): void => {
      cleanup();
      resolve();
    };
    const failed = (): void => {
      cleanup();
      reject(
        new Error("This browser cannot decode the replay. Export the original recording instead."),
      );
    };
    const aborted = (): void => {
      cleanup();
      reject(new DOMException("Cancelled", "AbortError"));
    };
    const timeout = setTimeout(() => {
      cleanup();
      reject(new Error("The replay took too long to decode. Try a shorter recording."));
    }, 15_000);
    video.addEventListener(event, ready, { once: true });
    video.addEventListener("error", failed, { once: true });
    signal?.addEventListener("abort", aborted, { once: true });
    if (signal?.aborted) aborted();
  });
}

/** Uniform candidates, explicitly not estimated camera poses or reconstructed geometry. */
export async function extractReplayFrames(
  blob: Blob,
  options: {
    count?: number;
    durationMs?: number;
    signal?: AbortSignal;
    onProgress?: (done: number, total: number) => void;
  } = {},
): Promise<ReconstructionSource[]> {
  const video = document.createElement("video");
  const url = URL.createObjectURL(blob);
  video.muted = true;
  video.preload = "auto";
  video.playsInline = true;
  try {
    const ready = waitFor(video, "loadeddata", options.signal);
    video.src = url;
    await ready;
    const duration = Number.isFinite(video.duration)
      ? video.duration
      : (options.durationMs ?? 0) / 1000;
    if (!(duration > 0))
      throw new Error("Replay duration is unavailable. Export the original recording instead.");
    const count = Math.min(120, Math.max(2, Math.floor(options.count ?? 24)));
    const canvas = document.createElement("canvas");
    const scale = Math.min(1, 1600 / Math.max(video.videoWidth, video.videoHeight));
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
    const context = canvas.getContext("2d");
    if (!context) throw new Error("This browser cannot extract replay frames.");
    const frames: ReconstructionSource[] = [];
    for (let index = 0; index < count; index++) {
      if (options.signal?.aborted) throw new DOMException("Cancelled", "AbortError");
      const time = (index / (count - 1)) * Math.max(0, duration - 0.1);
      if (Math.abs(video.currentTime - time) > 0.001) {
        const sought = waitFor(video, "seeked", options.signal);
        video.currentTime = time;
        await sought;
      }
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const frame = await new Promise<Blob>((resolve, reject) =>
        canvas.toBlob(
          (value) =>
            value ? resolve(value) : reject(new Error("Could not encode a replay frame.")),
          "image/jpeg",
          0.92,
        ),
      );
      frames.push({
        id: crypto.randomUUID(),
        blob: frame,
        name: `Frame ${index + 1}`,
        timestampMs: Math.round(time * 1000),
      });
      options.onProgress?.(index + 1, count);
    }
    return frames;
  } finally {
    video.removeAttribute("src");
    video.load();
    URL.revokeObjectURL(url);
  }
}
