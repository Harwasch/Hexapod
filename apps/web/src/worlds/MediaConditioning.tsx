import { useEffect, useRef, useState } from "react";
import { Film, Loader2, Sparkles } from "lucide-react";
import type { MediaAsset, WorldProject } from "./core/types";
import { worldStore } from "./core/storage";
import { extractReplayFrames } from "./reconstruction/frames";
import { dataImageBlob, imageData, postIntelligence } from "./intelligence/client";

export function MediaConditioning({
  project,
  assets,
  serverUrl,
  onChange,
}: {
  project: WorldProject;
  assets: MediaAsset[];
  serverUrl: string;
  onChange: (project: WorldProject) => void;
}) {
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const active = useRef<AbortController | null>(null);
  const sourceKey = project.assetIds.join(":");
  useEffect(
    () => () => active.current?.abort(),
    [project.id, project.modelId, project.prompt, sourceKey, serverUrl],
  );
  const images = assets.filter((asset) => asset.kind === "image");
  const video = assets.find((asset) => asset.kind === "video");
  async function prepare(kind: "video" | "image") {
    if (busy || (kind === "image" && !consent)) return;
    const controller = new AbortController();
    active.current = controller;
    setBusy(true);
    setError("");
    setMessage("");
    try {
      let image: Blob;
      if (kind === "video") {
        if (!video) throw new Error("Choose a video reference first.");
        const blob = await worldStore.getBlob(video.id);
        if (!blob) throw new Error("This video is missing from local storage.");
        const frames = await extractReplayFrames(blob, { count: 2, signal: controller.signal });
        const last = frames.at(-1);
        if (!last) throw new Error("The video has no decodable frames.");
        image = last.blob;
      } else {
        const references: string[] = [];
        for (const asset of images.slice(0, 6)) {
          const blob = await worldStore.getBlob(asset.id);
          if (!blob) throw new Error("A reference image is missing from local storage.");
          references.push(await imageData(blob));
        }
        controller.signal.throwIfAborted();
        const result = await postIntelligence<{ image: string }>(
          serverUrl,
          "/references/generate",
          { prompt: project.prompt, images: references },
          controller.signal,
        );
        image = dataImageBlob(result.image);
      }
      controller.signal.throwIfAborted();
      const asset = await worldStore.saveAsset(image, `${project.name} starting frame`, "image");
      controller.signal.throwIfAborted();
      onChange({
        ...project,
        assetIds: [asset.id],
        referencePreparation: {
          method: kind === "video" ? "last-video-frame" : "image-synthesis",
          sourceAssetIds: [...project.assetIds],
        },
        updatedAt: Date.now(),
      });
      setMessage(
        kind === "video"
          ? "Starting from the video's last frame. Earlier motion and audio are not passed to the world model."
          : "Generated starting frame added. Your original references remain in the local library.",
      );
    } catch (failure) {
      if (!controller.signal.aborted)
        setError(failure instanceof Error ? failure.message : String(failure));
    } finally {
      if (active.current === controller) {
        active.current = null;
        setBusy(false);
      }
    }
  }
  return (
    <div className="w-reference-preparation">
      <h3>Prepare a starting frame</h3>
      <p className="w-muted">
        Use a generated image or the last frame of a video with image-conditioned models. This
        preparation happens before the world session.
      </p>
      {video && (
        <button className="w-btn w-btn-quiet" disabled={busy} onClick={() => void prepare("video")}>
          <Film size={14} />
          Continue visually from the last video frame
        </button>
      )}
      <label className="w-check-label">
        <input
          type="checkbox"
          checked={consent}
          onChange={(event) => {
            setConsent(event.target.checked);
            if (!event.target.checked) active.current?.abort();
          }}
        />
        <span>Send the prompt and up to six reference images to my configured image service.</span>
      </label>
      <button
        className="w-btn w-btn-quiet"
        disabled={busy || !consent || !project.prompt.trim()}
        onClick={() => void prepare("image")}
      >
        {busy ? <Loader2 className="w-spin" size={14} /> : <Sparkles size={14} />}{" "}
        {images.length ? "Compose references into a starting frame" : "Generate a starting image"}
      </button>
      {project.referencePreparation && (
        <p className="w-muted">
          Prepared input:{" "}
          {project.referencePreparation.method === "last-video-frame"
            ? "visual continuation from a video frame"
            : "image synthesis"}
          .
        </p>
      )}
      {message && (
        <p className="w-muted" role="status">
          {message}
        </p>
      )}
      {error && (
        <p className="w-inline-error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
