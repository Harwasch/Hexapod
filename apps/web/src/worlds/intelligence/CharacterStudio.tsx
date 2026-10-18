import { useEffect, useRef, useState } from "react";
import { Check, Download, ImagePlus, Loader2, Sparkles } from "lucide-react";
import { worldStore } from "../core/storage";
import type { Character } from "../core/types";
import {
  dataImageBlob,
  getIntelligenceStatus,
  imageData,
  postIntelligence,
  type AppearancePackage,
  type ConsistencyAssessment,
  type IntelligenceStatus,
} from "./client";
import "./intelligence.css";
export interface CharacterStudioProps {
  serverUrl: string;
  character: Character;
  onSave: (character: Character) => Promise<void>;
  onUseReference?: (assetId: string, prompt: string) => void;
  initialScene?: string;
}
export function CharacterStudio({
  serverUrl,
  character,
  onSave,
  onUseReference,
  initialScene = "",
}: CharacterStudioProps) {
  const [status, setStatus] = useState<IntelligenceStatus>();
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [appearance, setAppearance] = useState<AppearancePackage>();
  const [assessment, setAssessment] = useState<ConsistencyAssessment>();
  const [scene, setScene] = useState(initialScene);
  const [candidate, setCandidate] = useState<File>();
  const [generated, setGenerated] = useState<{ url: string; assetId: string; kind: string }>();
  const controller = useRef<AbortController | null>(null);
  const current = useRef(character);
  useEffect(() => {
    current.current = character;
  }, [character]);
  useEffect(() => {
    const abort = new AbortController();
    void getIntelligenceStatus(serverUrl, abort.signal)
      .then(setStatus)
      .catch((failure: unknown) => {
        if (!abort.signal.aborted)
          setError(failure instanceof Error ? failure.message : "AI services unavailable.");
      });
    return () => {
      abort.abort();
      controller.current?.abort();
    };
  }, [serverUrl]);
  useEffect(
    () => () => {
      if (generated?.url) URL.revokeObjectURL(generated.url);
    },
    [generated],
  );
  async function references(required = true) {
    const records = await worldStore.list("assets");
    const blobs: Blob[] = [];
    for (const id of current.current.assetIds) {
      if (records.find((a) => a.id === id)?.kind !== "image") continue;
      const blob = await worldStore.getBlob(id);
      if (blob) blobs.push(blob);
      if (blobs.length === 6) break;
    }
    if (required && !blobs.length)
      throw new Error("Add at least one reference photo to this character first.");
    return Promise.all(blobs.map(imageData));
  }
  async function run(task: "analyze" | "portrait" | "full-body" | "scene" | "evaluate") {
    if (busy || !consent) return;
    setBusy(task);
    setError("");
    setMessage("");
    const abort = new AbortController();
    controller.current = abort;
    try {
      const images = await references(task === "analyze" || task === "evaluate");
      abort.signal.throwIfAborted();
      if (task === "analyze") {
        const result = await postIntelligence<AppearancePackage>(
          serverUrl,
          "/characters/analyze",
          { description: current.current.description.slice(0, 4000), images },
          abort.signal,
        );
        setAppearance(result);
      } else if (task === "evaluate") {
        if (!candidate) throw new Error("Select a generated candidate image to compare.");
        const result = await postIntelligence<ConsistencyAssessment>(
          serverUrl,
          "/characters/evaluate",
          {
            description: current.current.description.slice(0, 4000),
            references: images.slice(0, 4),
            candidates: [await imageData(candidate)],
          },
          abort.signal,
        );
        setAssessment(result);
      } else {
        const result = await postIntelligence<{ image: string; note: string }>(
          serverUrl,
          "/characters/synthesize",
          {
            description: (appearance?.conditioningPrompt ?? current.current.description).slice(
              0,
              4000,
            ),
            images,
            kind: task,
            scenePrompt: scene,
          },
          abort.signal,
        );
        abort.signal.throwIfAborted();
        const asset = await worldStore.saveAsset(
          dataImageBlob(result.image),
          `${current.current.name} ${task}.jpg`,
          "image",
        );
        const blob = await worldStore.getBlob(asset.id);
        if (!blob) throw new Error("Generated image could not be reopened.");
        setGenerated({ url: URL.createObjectURL(blob), assetId: asset.id, kind: task });
        setMessage(result.note);
      }
    } catch (failure) {
      if (!abort.signal.aborted)
        setError(failure instanceof Error ? failure.message : "Character processing failed.");
    } finally {
      if (controller.current === abort) {
        setBusy("");
        controller.current = null;
      }
    }
  }
  async function savePackage() {
    if (!appearance) return;
    try {
      const description = [
        appearance.appearance,
        `Clothing: ${appearance.clothing}`,
        `Distinguishing features: ${appearance.distinguishingFeatures.join("; ")}`,
        `Conditioning: ${appearance.conditioningPrompt}`,
        `Reference caveats: ${appearance.consistencyNotes.join("; ")}`,
      ].join("\n");
      await onSave({
        ...current.current,
        description,
        identityMethod: "reference-images",
        updatedAt: Date.now(),
      });
      setMessage("Appearance package saved to the character description.");
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not save character.");
    }
  }
  async function attachGenerated() {
    if (!generated) return;
    try {
      await onSave({
        ...current.current,
        assetIds: [...new Set([...current.current.assetIds, generated.assetId])],
        identityMethod: "reference-images",
        updatedAt: Date.now(),
      });
      setMessage("Generated reference added to this local character.");
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not attach reference.");
    }
  }
  const downloadAssessment = () => {
    if (!assessment) return;
    const url = URL.createObjectURL(
      new Blob(
        [
          JSON.stringify(
            { characterId: character.id, createdAt: new Date().toISOString(), ...assessment },
            null,
            2,
          ),
        ],
        { type: "application/json" },
      ),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = "character-appearance-assessment.json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return (
    <section className="wi-panel" aria-label="Character studio">
      <h3>
        <Sparkles size={16} />
        Character studio
      </h3>
      <p className="wi-muted">
        Analyze several references, synthesize a clean portrait, or place a character into a
        starting frame. These are image-conditioning tools; identity preservation is not guaranteed.
      </p>
      <label className="wi-check">
        <input
          type="checkbox"
          checked={consent}
          onChange={(e) => {
            setConsent(e.target.checked);
            if (!e.target.checked) controller.current?.abort();
          }}
        />
        Send selected reference pixels and descriptions to configured vision/image services. Local
        filenames and image metadata are stripped.
      </label>
      {!status?.visionConfigured && (
        <p className="wi-notice">
          Reference analysis requires WORLDS_VISION_BASE_URL and WORLDS_VISION_MODEL.
        </p>
      )}
      <button
        type="button"
        disabled={!consent || !!busy || !status?.visionConfigured}
        onClick={() => void run("analyze")}
      >
        <Sparkles size={14} />
        Analyze reference photos
      </button>
      {appearance && (
        <div className="wi-result">
          <p>{appearance.appearance}</p>
          <p>{appearance.clothing}</p>
          <ul>
            {appearance.distinguishingFeatures.map((f, i) => (
              <li key={i}>{f}</li>
            ))}
          </ul>
          {appearance.consistencyNotes.map((n, i) => (
            <p className="wi-muted" key={i}>
              {n}
            </p>
          ))}
          <button type="button" onClick={() => void savePackage()}>
            <Check size={14} />
            Save appearance package
          </button>
        </div>
      )}
      {!status?.imageConfigured && (
        <p className="wi-notice">
          Portrait and scene generation require a compatible server-side WORLDS_IMAGE_BASE_URL /
          WORLDS_IMAGE_MODEL image-edit endpoint.
        </p>
      )}
      <div className="wi-row">
        <button
          type="button"
          disabled={!consent || !!busy || !status?.imageConfigured}
          onClick={() => void run("portrait")}
        >
          <ImagePlus size={14} />
          Generate portrait
        </button>
        <button
          type="button"
          disabled={!consent || !!busy || !status?.imageConfigured}
          onClick={() => void run("full-body")}
        >
          Full-body reference
        </button>
      </div>
      <label className="wi-field">
        Starting world scene
        <textarea
          maxLength={6000}
          value={scene}
          onChange={(e) => setScene(e.target.value)}
          placeholder="The character stands on a rain-soaked jungle platform at dusk…"
        />
      </label>
      <button
        type="button"
        disabled={!consent || !!busy || !scene.trim() || !status?.imageConfigured}
        onClick={() => void run("scene")}
      >
        Generate character-conditioned scene
      </button>
      {generated && (
        <div className="wi-result">
          <img
            className="wi-reference"
            src={generated.url}
            alt={`Generated ${generated.kind} for ${character.name}`}
          />
          <div className="wi-row">
            <a
              className="wi-button"
              href={generated.url}
              download={`${character.name}-${generated.kind}.jpg`}
            >
              <Download size={14} />
              Download image
            </a>
            <button type="button" onClick={() => void attachGenerated()}>
              Add to character
            </button>
            {generated.kind === "scene" && onUseReference && (
              <button type="button" onClick={() => onUseReference(generated.assetId, scene)}>
                Use as world input
              </button>
            )}
          </div>
          <p className="wi-muted">
            Use the scene as a single reference image on image-conditioned world models. This does
            not add a native identity channel.
          </p>
        </div>
      )}
      <label className="wi-field">
        Evaluate a generated character image
        <input
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={(e) => setCandidate(e.target.files?.[0])}
        />
      </label>
      <button
        type="button"
        disabled={!consent || !!busy || !candidate || !status?.visionConfigured}
        onClick={() => void run("evaluate")}
      >
        Compare visible appearance
      </button>
      {assessment && (
        <div className="wi-result">
          <strong>
            Appearance consistency: {Math.round(assessment.score * 100)}% · AI confidence{" "}
            {Math.round(assessment.confidence * 100)}%
          </strong>
          <p className="wi-muted">{assessment.note}</p>
          <ul>
            {assessment.evidence.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
          {assessment.differences.map((e, i) => (
            <p key={i}>{e}</p>
          ))}
          <button type="button" onClick={downloadAssessment}>
            <Download size={14} />
            Export assessment with evidence
          </button>
        </div>
      )}
      {busy && (
        <p role="status">
          <Loader2 className="w-spin" size={14} />{" "}
          {busy === "analyze"
            ? "Analyzing references…"
            : busy === "evaluate"
              ? "Comparing appearance…"
              : "Waiting for the configured image model…"}
        </p>
      )}
      {error && (
        <p className="wi-error" role="alert">
          {error}
        </p>
      )}
      {message && (
        <p className="wi-muted" role="status">
          {message}
        </p>
      )}
    </section>
  );
}
