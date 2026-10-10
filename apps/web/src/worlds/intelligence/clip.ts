/** Local replay trimming. Re-encodes playback; no media upload and no microphone access. */
export async function extractReplayClip(
  source: Blob,
  startSeconds: number,
  endSeconds: number,
  signal: AbortSignal,
): Promise<Blob> {
  if (
    !Number.isFinite(startSeconds) ||
    !Number.isFinite(endSeconds) ||
    startSeconds < 0 ||
    endSeconds - startSeconds < 2 ||
    endSeconds - startSeconds > 120
  )
    throw new Error("Select a clip between 2 and 120 seconds long.");
  signal.throwIfAborted();
  if (typeof MediaRecorder === "undefined")
    throw new Error(
      "This browser cannot encode replay clips. Export the clip markers and original replay instead.",
    );
  const video = document.createElement("video") as HTMLVideoElement & {
    captureStream?: () => MediaStream;
  };
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  // Keep playback attached so browsers do not suspend decoded frames from detached media.
  video.setAttribute("aria-hidden", "true");
  video.style.cssText =
    "position:fixed;width:2px;height:2px;bottom:0;right:0;opacity:.01;pointer-events:none";
  document.body.appendChild(video);
  const url = URL.createObjectURL(source);
  let stream: MediaStream | undefined;
  let recorder: MediaRecorder | undefined;
  let frame = 0;
  let watchdog = 0;
  let boundary: (() => void) | undefined;
  let failRecording: ((message: string) => void) | undefined;
  const wait = (event: "loadeddata" | "seeked") =>
    new Promise<void>((resolve, reject) => {
      if (signal.aborted) {
        reject(new DOMException("Aborted", "AbortError"));
        return;
      }
      const clear = () => {
        video.removeEventListener(event, done);
        video.removeEventListener("error", failed);
        signal.removeEventListener("abort", aborted);
        window.clearTimeout(timeout);
      };
      const done = () => {
        clear();
        resolve();
      };
      const failed = () => {
        clear();
        reject(new Error(`Could not decode the replay clip. ${video.error?.message ?? ""}`.trim()));
      };
      const aborted = () => {
        clear();
        reject(new DOMException("Aborted", "AbortError"));
      };
      const timeout = window.setTimeout(failed, 15_000);
      video.addEventListener(event, done, { once: true });
      video.addEventListener("error", failed, { once: true });
      signal.addEventListener("abort", aborted, { once: true });
    });
  try {
    const loaded = wait("loadeddata");
    video.src = url;
    await loaded;
    if (Number.isFinite(video.duration) && endSeconds > video.duration + 0.1)
      throw new Error("The highlight extends beyond this replay.");
    if (startSeconds > 0.001) {
      const sought = wait("seeked");
      video.currentTime = startSeconds;
      await sought;
    }
    signal.throwIfAborted();
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d", { alpha: false });
    if (!context || !canvas.captureStream)
      throw new Error("This browser cannot capture replay frames.");
    context.drawImage(video, 0, 0);
    const capture = canvas.captureStream(0);
    const track = capture.getVideoTracks()[0] as CanvasCaptureMediaStreamTrack;
    stream = capture;
    const originalTracks = video.captureStream?.();
    originalTracks?.getAudioTracks().forEach((audio) => capture.addTrack(audio));
    originalTracks?.getVideoTracks().forEach((videoTrack) => videoTrack.stop());
    const mimeType = [
      "video/webm;codecs=vp8",
      "video/webm;codecs=vp9,opus",
      "video/webm;codecs=vp8,opus",
      "video/webm",
      "video/mp4",
    ].find((type) => MediaRecorder.isTypeSupported(type));
    if (!mimeType) throw new Error("This browser has no compatible video encoder.");
    const output = new MediaRecorder(capture, { mimeType, videoBitsPerSecond: 6_000_000 });
    recorder = output;
    const chunks: Blob[] = [];
    const complete = new Promise<Blob>((resolve, reject) => {
      let settled = false;
      const cleanup = () => {
        signal.removeEventListener("abort", abort);
      };
      const abort = () => {
        if (settled) return;
        settled = true;
        cleanup();
        if (output.state !== "inactive") output.stop();
        reject(new DOMException("Aborted", "AbortError"));
      };
      output.ondataavailable = (e) => {
        if (e.data.size) chunks.push(e.data);
      };
      failRecording = (message) => {
        if (settled) return;
        settled = true;
        cleanup();
        if (output.state !== "inactive") output.stop();
        reject(new Error(message));
      };
      output.onerror = () => failRecording?.("Replay clip encoding failed.");
      output.onstop = () => {
        if (settled) return;
        settled = true;
        cleanup();
        const blob = new Blob(chunks, { type: output.mimeType });
        if (blob.size) resolve(blob);
        else reject(new Error("The browser did not record any clip frames."));
      };
      signal.addEventListener("abort", abort, { once: true });
    });
    // Attach a rejection handler before playback so a play() failure cannot leave an unhandled promise.
    void complete.catch(() => undefined);
    output.start(250);
    await video.play();
    const tick = () => {
      context.drawImage(video, 0, 0);
      track.requestFrame();
      if (video.currentTime >= endSeconds || video.ended) {
        video.pause();
        if (output.state !== "inactive") output.stop();
      } else frame = requestAnimationFrame(tick);
    };
    boundary = () => {
      if (video.currentTime >= endSeconds || video.ended) {
        video.pause();
        if (output.state !== "inactive") output.stop();
      }
    };
    video.addEventListener("timeupdate", boundary);
    frame = requestAnimationFrame(tick);
    watchdog = window.setTimeout(
      () => {
        video.pause();
        failRecording?.("Replay playback stalled before the full clip could be recorded.");
      },
      (endSeconds - startSeconds + 10) * 1000,
    );
    return await complete;
  } finally {
    cancelAnimationFrame(frame);
    if (boundary) video.removeEventListener("timeupdate", boundary);
    window.clearTimeout(watchdog);
    video.pause();
    if (recorder && recorder.state !== "inactive") recorder.stop();
    stream?.getTracks().forEach((track) => track.stop());
    video.removeAttribute("src");
    video.load();
    video.remove();
    URL.revokeObjectURL(url);
  }
}
