import { useEffect, useRef, useState } from "react";
import { Download, Sparkles } from "lucide-react";
import { imageData, postIntelligence, type HighlightsResult } from "./client";
import "./intelligence.css";
import { extractReplayClip } from "./clip";
export interface ReplayHighlightsProps {
  serverUrl: string;
  video: Blob;
  durationSeconds: number;
  prompt?: string;
  onSeek?: (seconds: number) => void;
}
async function sampleFrames(blob: Blob, duration: number, signal: AbortSignal) {
  signal.throwIfAborted();
  const video = document.createElement("video");
  video.muted = true;
  video.preload = "auto";
  const url = URL.createObjectURL(blob);
  video.src = url;
  const wait = (name: "loadeddata" | "seeked") =>
    new Promise<void>((resolve, reject) => {
      if (signal.aborted) {
        reject(new DOMException("Aborted", "AbortError"));
        return;
      }
      const done = () => {
        cleanup();
        resolve();
      };
      const failed = () => {
        cleanup();
        reject(new Error("Replay frame could not be decoded."));
      };
      const abort = () => {
        cleanup();
        reject(new DOMException("Aborted", "AbortError"));
      };
      const timeout = setTimeout(failed, 15_000);
      const cleanup = () => {
        clearTimeout(timeout);
        video.removeEventListener(name, done);
        video.removeEventListener("error", failed);
        signal.removeEventListener("abort", abort);
      };
      video.addEventListener(name, done, { once: true });
      video.addEventListener("error", failed, { once: true });
      signal.addEventListener("abort", abort, { once: true });
    });
  try {
    if (video.readyState < 2) await wait("loadeddata");
    if (!video.videoWidth || !video.videoHeight)
      throw new Error("The replay has no decodable video frames.");
    const canvas = document.createElement("canvas");
    canvas.width = 640;
    canvas.height = Math.max(1, Math.round((640 * video.videoHeight) / video.videoWidth));
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Replay sampling is unavailable.");
    const samples = [];
    for (let i = 0; i < 6; i++) {
      signal.throwIfAborted();
      const atSeconds = Math.max(0.05, Math.min(duration - 0.05, ((i + 0.5) * duration) / 6));
      if (Math.abs(video.currentTime - atSeconds) > 0.001) {
        const ready = wait("seeked");
        video.currentTime = atSeconds;
        await ready;
      }
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      const frame = await new Promise<Blob>((resolve, reject) =>
        canvas.toBlob(
          (b) => (b ? resolve(b) : reject(new Error("Could not capture replay frame."))),
          "image/jpeg",
          0.8,
        ),
      );
      samples.push({ atSeconds, image: await imageData(frame) });
    }
    return samples;
  } finally {
    video.removeAttribute("src");
    video.load();
    URL.revokeObjectURL(url);
  }
}
export function ReplayHighlights({
  serverUrl,
  video,
  durationSeconds,
  prompt = "",
  onSeek,
}: ReplayHighlightsProps) {
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<HighlightsResult>();
  const [error, setError] = useState("");
  const [clipBusy, setClipBusy] = useState<number | null>(null);
  const request = useRef<AbortController | null>(null);
  useEffect(() => () => request.current?.abort(), []);
  const find = async () => {
    if (!consent || busy) return;
    setBusy(true);
    setError("");
    const controller = new AbortController();
    request.current = controller;
    try {
      if (durationSeconds < 0.2 || !Number.isFinite(durationSeconds))
        throw new Error("A replay duration is required.");
      const samples = await sampleFrames(video, durationSeconds, controller.signal);
      setResult(
        await postIntelligence<HighlightsResult>(
          serverUrl,
          "/intelligence/highlights",
          { samples, durationSeconds, prompt },
          controller.signal,
        ),
      );
    } catch (failure) {
      if (!controller.signal.aborted)
        setError(failure instanceof Error ? failure.message : "Highlight analysis failed.");
    } finally {
      setBusy(false);
    }
  };
  const downloadClip = async (index: number) => {
    const highlight = result?.highlights[index];
    if (!highlight || clipBusy !== null || busy) return;
    setClipBusy(index);
    setError("");
    const controller = new AbortController();
    request.current = controller;
    try {
      const clip = await extractReplayClip(
        video,
        highlight.startSeconds,
        highlight.endSeconds,
        controller.signal,
      );
      const url = URL.createObjectURL(clip);
      const link = document.createElement("a");
      link.href = url;
      link.download = `highlight-${index + 1}.${clip.type.includes("mp4") ? "mp4" : "webm"}`;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (failure) {
      if (!controller.signal.aborted)
        setError(failure instanceof Error ? failure.message : "Clip export failed.");
    } finally {
      setClipBusy(null);
    }
  };
  const download = () => {
    if (!result) return;
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = "replay-highlight-markers.json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return (
    <section className="wi-panel">
      <h3>
        <Sparkles size={16} />
        Find the moments
      </h3>
      <label className="wi-check">
        <input
          type="checkbox"
          checked={consent}
          onChange={(e) => {
            setConsent(e.target.checked);
            if (!e.target.checked) request.current?.abort();
          }}
        />
        Send six sampled replay frames to the configured vision model. The full video stays local.
      </label>
      <button
        type="button"
        disabled={!consent || busy || clipBusy !== null}
        onClick={() => void find()}
      >
        {busy ? "Reviewing replay frames…" : "Suggest highlights"}
      </button>
      {error && (
        <p className="wi-error" role="alert">
          {error}
        </p>
      )}
      {result && (
        <>
          <p className="wi-muted">
            {result.note} Clip export re-encodes locally at playback speed, from two seconds to two
            minutes per clip.
          </p>
          {result.highlights.length === 0 && (
            <p>No distinct highlights were identified in these samples.</p>
          )}
          {result.highlights.map((h, i) => (
            <div className="wi-result" key={i}>
              <strong>
                {h.title} · {h.startSeconds.toFixed(1)}–{h.endSeconds.toFixed(1)}s
              </strong>
              <p>{h.evidence}</p>
              <button
                type="button"
                disabled={busy || clipBusy !== null}
                onClick={() => void downloadClip(i)}
              >
                <Download size={14} />
                {clipBusy === i ? "Encoding clip locally…" : "Download this clip"}
              </button>
              {onSeek && (
                <button type="button" onClick={() => onSeek(h.startSeconds)}>
                  Review this moment
                </button>
              )}
            </div>
          ))}
          <button type="button" onClick={download}>
            <Download size={14} />
            Export clip markers
          </button>
        </>
      )}
    </section>
  );
}
