import { useEffect, useRef, useState } from "react";
import { Activity, Download, FlaskConical, Play, Square } from "lucide-react";
import { createWorldApi } from "../core/api";
import { MODELS, getModel } from "../core/catalog";
import { parseTrajectory } from "../core/controls";
import { downloadBlob, worldStore } from "../core/storage";
import type { BenchmarkResult, Character, ControlEvent, WorldProject } from "../core/types";
import { runExperiment } from "./runner";
import {
  isExperimentResult,
  type ExperimentPlan,
  type ExperimentProgress,
  type ExperimentResult,
  type ExperimentTarget,
} from "./types";
import "./experiments.css";
export interface ExperimentLabProps {
  projects: WorldProject[];
  results: BenchmarkResult[];
  characters?: Character[];
  serverUrl: string;
  onSaved?: () => void;
}
function useAssetUrl(id?: string) {
  const [asset, setAsset] = useState<{ id: string; url: string }>();
  useEffect(() => {
    let current: string | undefined;
    let active = true;
    if (id)
      void worldStore.getBlob(id).then((blob) => {
        if (blob && active) {
          current = URL.createObjectURL(blob);
          setAsset({ id, url: current });
        }
      });
    return () => {
      active = false;
      if (current) URL.revokeObjectURL(current);
    };
  }, [id]);
  return asset?.id === id ? asset?.url : undefined;
}
const percentage = (value?: number) => (value === undefined ? "—" : `${(value * 100).toFixed(1)}%`);
function ResultCard({ result, onSaved }: { result: ExperimentResult; onSaved?: () => void }) {
  const video = useAssetUrl(result.videoAssetId);
  const image = useAssetUrl(result.observations.find((frame) => frame.assetId)?.assetId);
  const [ratings, setRatings] = useState(result.ratings ?? {});
  const [error, setError] = useState("");
  const saveRating = async (key: keyof NonNullable<BenchmarkResult["ratings"]>, value: string) => {
    const next = Object.fromEntries(
      Object.entries(ratings).filter(([name]) => name !== key),
    ) as NonNullable<BenchmarkResult["ratings"]>;
    if (value) next[key] = Number(value);
    try {
      await worldStore.put("benchmarks", { ...result, ratings: next });
      setRatings(next);
      onSaved?.();
    } catch (cause) {
      setError(String(cause));
    }
  };
  const motion = result.observations.filter((frame) => frame.motionX !== undefined);
  const mean = (key: "motionX" | "motionY" | "pixelChange" | "motionConfidence") =>
    motion.length
      ? motion.reduce((sum, frame) => sum + (frame[key] ?? 0), 0) / motion.length
      : undefined;
  return (
    <article className="we-result">
      <header>
        <div>
          <strong>{result.targetLabel}</strong>
          <span>
            {result.status} · seed {result.seed}
          </span>
        </div>
        <button
          className="we-icon"
          title="Export measured result"
          onClick={() =>
            downloadBlob(
              new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }),
              `worlds-result-${result.id}.json`,
            )
          }
        >
          <Download size={16} />
        </button>
      </header>
      <div className="we-result-media">
        {video ? (
          <video src={video} controls muted preload="metadata" />
        ) : image ? (
          <img src={image} alt={`Actual generated frame from ${result.targetLabel}`} />
        ) : (
          <span>No generated output captured</span>
        )}
      </div>
      <dl>
        <div>
          <dt>Generated FPS</dt>
          <dd>{result.generatedFPS?.toFixed(2) ?? "Not reported"}</dd>
        </div>
        <div>
          <dt>Unique sampled FPS</dt>
          <dd>{result.deliveredFPS.toFixed(2)}</dd>
        </div>
        <div>
          <dt>First frame</dt>
          <dd>
            {result.firstFrameMs === undefined
              ? "—"
              : `${(result.firstFrameMs / 1000).toFixed(1)} s`}
          </dd>
        </div>
        <div>
          <dt>Control round trip</dt>
          <dd>{result.latencyMs?.toFixed(0) ?? "—"} ms</dd>
        </div>
        <div>
          <dt>Estimated cost</dt>
          <dd>${(result.estimatedCostUSD ?? 0).toFixed(4)}</dd>
        </div>
        <div>
          <dt>Accepted actions</dt>
          <dd>
            {result.actionObservations.filter((action) => action.accepted).length}/
            {result.actionObservations.length}
          </dd>
        </div>
        <div>
          <dt>Frame similarity, first → last</dt>
          <dd>{percentage(result.returnFrameSimilarity)}</dd>
        </div>
        <div>
          <dt>Mean pixel change</dt>
          <dd>{percentage(mean("pixelChange"))}</dd>
        </div>
        <div>
          <dt>Estimated image shift</dt>
          <dd>
            {mean("motionX")?.toFixed(1) ?? "—"}, {mean("motionY")?.toFixed(1) ?? "—"} px
          </dd>
        </div>
        <div>
          <dt>Translation fit improvement</dt>
          <dd>{percentage(mean("motionConfidence"))}</dd>
        </div>
      </dl>
      {result.characterEvaluation && (
        <section className="we-character-scores">
          <h4>
            {result.characterEvaluation.characterName} · {result.characterEvaluation.view}
          </h4>
          <p>
            {result.characterEvaluation.source === "vision-llm"
              ? `Observed appearance score ${percentage(result.characterEvaluation.meanScore)} · confidence ${percentage(result.characterEvaluation.meanConfidence)} · ${result.characterEvaluation.samples.length} evaluated samples`
              : "Automatic appearance assessment not performed."}
          </p>
          <p>{result.characterEvaluation.conditioningMethod}</p>
          {result.characterEvaluation.samples.map((sample) => (
            <details key={sample.assetId}>
              <summary>
                {(sample.timestampMs / 1000).toFixed(1)} s · appearance {percentage(sample.score)} ·
                confidence {percentage(sample.confidence)}
              </summary>
              <p>{sample.evidence.join(" ")}</p>
              {!!sample.differences.length && <p>Differences: {sample.differences.join(" ")}</p>}
            </details>
          ))}
          {result.characterEvaluation.errors.map((message, index) => (
            <p className="we-error" key={index}>
              {message}
            </p>
          ))}
          <small>
            Qualitative visible design assessment, not face recognition, identity verification, or
            proof of persistent character state.
          </small>
        </section>
      )}
      <details>
        <summary>Human ratings and event evidence</summary>
        <div className="we-ratings">
          {(
            [
              "visualFidelity",
              "actionAdherence",
              "temporalStability",
              "spatialConsistency",
              "characterConsistency",
            ] as const
          ).map((key) => (
            <label key={key}>
              {key.replace(/([A-Z])/g, " $1")}
              <select
                aria-label={`${result.targetLabel} ${key}`}
                value={ratings[key] ?? ""}
                onChange={(event) => void saveRating(key, event.target.value)}
              >
                <option value="">Unrated</option>
                {[1, 2, 3, 4, 5].map((value) => (
                  <option key={value}>{value}</option>
                ))}
              </select>
            </label>
          ))}
        </div>
        <p>
          Shift uses block matching on 64 × 40 grayscale frames. Pixel change and first/last
          similarity do not establish camera pose, geometry, identity, or adherence. Generated FPS
          is reported by the worker; sampled capture omits native audio.
        </p>
        <ol>
          {result.actionObservations.map((action) => (
            <li key={action.eventId}>
              {(action.dispatchedMs / 1000).toFixed(2)} s ·{" "}
              {action.accepted ? "accepted" : action.error} · {action.latencyMs.toFixed(0)} ms
            </li>
          ))}
        </ol>
        <small>
          Input fingerprint: {result.inputFingerprint}
          <br />
          Effective prompt:{" "}
          {result.effectivePrompt?.length
            ? result.effectivePrompt
            : "Text unsupported by this adapter"}
          <br />
          Effective quality: {result.effectiveQuality ?? "balanced"}
          <br />
          Pricing: {result.priceSource ?? "unavailable"} · billed cost not reported.
        </small>
      </details>
      {[...result.errors, ...result.cleanupErrors, error].filter(Boolean).map((message, index) => (
        <p className="we-error" key={index}>
          {message}
        </p>
      ))}
    </article>
  );
}
export default function ExperimentLab({
  projects,
  results,
  characters = [],
  serverUrl,
  onSaved,
}: ExperimentLabProps) {
  const [projectId, setProjectId] = useState("");
  const [selected, setSelected] = useState<string[]>(["astronex-world"]);
  const [mode, setMode] = useState<ExperimentPlan["mode"]>("compare");
  const [seed, setSeed] = useState("42");
  const [duration, setDuration] = useState("30");
  const [wall, setWall] = useState("600");
  const [budget, setBudget] = useState("1");
  const [events, setEvents] = useState<ControlEvent[]>([]);
  const [progress, setProgress] = useState<ExperimentProgress>();
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [localResults, setLocalResults] = useState<ExperimentResult[]>([]);
  const [consent, setConsent] = useState(false);
  const [characterId, setCharacterId] = useState("");
  const [visionConsent, setVisionConsent] = useState(false);
  const [characterViews, setCharacterViews] = useState<("front" | "side" | "environment")[]>([
    "front",
    "side",
    "environment",
  ]);
  const character = characters.find((value) => value.id === characterId);
  const mounted = useRef(true);
  const resultGrid = useRef<HTMLDivElement>(null);
  const controller = useRef<AbortController | undefined>(undefined);
  const project = projects.find((value) => value.id === projectId);
  const availableProjects = projects.filter((value) => !value.previewOnly);
  useEffect(() => {
    mounted.current = true;
    const cancel = () =>
      controller.current?.abort(new Error("Experiment view closed; cleaning up compute."));
    window.addEventListener("pagehide", cancel);
    return () => {
      mounted.current = false;
      window.removeEventListener("pagehide", cancel);
      cancel();
    };
  }, []);
  const all = [...localResults, ...results.filter(isExperimentResult)].filter(
    (value, index, list) => list.findIndex((item) => item.id === value.id) === index,
  );
  const start = async () => {
    if (!project || !consent || (mode === "character-consistency" && !character)) return;
    setError("");
    setRunning(true);
    const cancellation = new AbortController();
    controller.current = cancellation;
    let targets: ExperimentTarget[] = selected.map((modelId) => ({
      modelId,
      providerId: project.providerId,
      label: getModel(modelId).name,
    }));
    if (mode === "character-consistency")
      targets = selected.flatMap((modelId) =>
        characterViews.map((characterView) => ({
          modelId,
          providerId: project.providerId,
          label: `${getModel(modelId).name} · ${characterView}`,
          characterView,
        })),
      );
    if (mode === "action-discovery") {
      const model = selected[0] ? getModel(selected[0]) : undefined;
      if (!model) {
        setRunning(false);
        return;
      }
      targets = [
        { modelId: model.id, providerId: project.providerId, label: "Baseline · no controls" },
        ...model.nativeActions
          .filter((action) => action !== "stop")
          .slice(0, 5)
          .map((action) => ({
            modelId: model.id,
            providerId: project.providerId,
            label: `Probe · ${action}`,
            discoveryAction: action,
          })),
      ];
    }
    try {
      const completed = await runExperiment(
        {
          project,
          targets,
          seed: Number(seed),
          durationSeconds: Number(duration),
          maxWallTimeSeconds: Number(wall),
          maxEstimatedCostUSD: Number(budget),
          events: mode === "action-discovery" ? [] : events,
          mode,
          character: mode === "character-consistency" ? character : undefined,
          visionConsent: mode === "character-consistency" && visionConsent,
          evaluationFrames: 2,
        },
        { api: createWorldApi(serverUrl) },
        cancellation.signal,
        (value) => {
          if (mounted.current) setProgress(value);
        },
      );
      if (mounted.current) {
        setLocalResults((previous) => [...completed, ...previous]);
        onSaved?.();
      }
    } catch (cause) {
      if (mounted.current) setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      if (mounted.current) setRunning(false);
      controller.current = undefined;
    }
  };
  const importControls = async (file?: File) => {
    if (!file) return;
    try {
      if (file.size > 1_000_000) throw new Error("Control file exceeds 1 MB.");
      const trajectory = parseTrajectory(await file.text());
      setEvents(trajectory.events);
      if (trajectory.seed !== undefined) setSeed(String(trajectory.seed));
      setError("");
    } catch (cause) {
      setError(String(cause));
    }
  };
  return (
    <section className="we-lab" aria-label="Executable model experiments">
      <div className="we-heading">
        <div>
          <span className="we-eyebrow">
            <FlaskConical size={14} /> WORLD MODEL EXPERIMENTS
          </span>
          <h2>Same idea. Measurable differences.</h2>
          <p>
            Run real model sessions sequentially with one prompt, seed and control trajectory.
            Outputs and measurements stay in your browser library.
          </p>
        </div>
        <Activity size={32} />
      </div>
      <div className="we-controls">
        <label>
          Source world
          <select
            value={projectId}
            onChange={(event) => setProjectId(event.target.value)}
            disabled={running}
          >
            <option value="">Choose a saved world</option>
            {availableProjects.map((world) => (
              <option key={world.id} value={world.id}>
                {world.name} · {world.providerId}
              </option>
            ))}
          </select>
        </label>
        <label>
          Experiment
          <select
            aria-label="Experiment mode"
            value={mode}
            onChange={(event) => setMode(event.target.value as typeof mode)}
            disabled={running}
          >
            <option value="compare">Compare models</option>
            <option value="action-discovery">Discover action effects</option>
            <option value="return-to-view">Return-to-view experiment</option>
            <option value="character-consistency">Character consistency</option>
          </select>
        </label>
        <label>
          Seed
          <input
            type="number"
            min="0"
            max="4294967295"
            value={seed}
            onChange={(event) => setSeed(event.target.value)}
            disabled={running}
          />
        </label>
        <label>
          Capture seconds
          <input
            type="number"
            min="5"
            max="300"
            value={duration}
            onChange={(event) => setDuration(event.target.value)}
            disabled={running}
          />
        </label>
        <label>
          Max seconds per run
          <input
            type="number"
            min="5"
            max="1800"
            value={wall}
            onChange={(event) => setWall(event.target.value)}
            disabled={running}
          />
        </label>
        <label>
          Suite estimate limit ($)
          <input
            type="number"
            min="0"
            max="100"
            step="0.25"
            value={budget}
            onChange={(event) => setBudget(event.target.value)}
            disabled={running}
          />
        </label>
      </div>
      {mode === "character-consistency" && (
        <div className="we-character-config">
          <label>
            Reusable character
            <select
              aria-label="Experiment character"
              value={characterId}
              disabled={running}
              onChange={(event) => {
                setCharacterId(event.target.value);
                setVisionConsent(false);
              }}
            >
              <option value="">Choose a saved character</option>
              {characters.map((value) => (
                <option key={value.id} value={value.id}>
                  {value.name}
                </option>
              ))}
            </select>
          </label>
          <div className="we-models">
            {(["front", "side", "environment"] as const).map((view) => (
              <label key={view}>
                <input
                  type="checkbox"
                  disabled={running}
                  checked={characterViews.includes(view)}
                  onChange={() =>
                    setCharacterViews((previous) =>
                      previous.includes(view)
                        ? previous.filter((item) => item !== view)
                        : [...previous, view],
                    )
                  }
                />
                {view === "environment" ? "New environment" : `${view} view`}
              </label>
            ))}
          </div>
          <p className="we-note">
            Each model/view gets an independent session and the same seed. The character description
            is appended only for text-capable adapters. Image-capable adapters start from the
            character reference; it replaces the world's initial image. This is reference
            conditioning, not guaranteed native identity preservation.
          </p>
          <label className="we-consent">
            <input
              type="checkbox"
              checked={visionConsent}
              disabled={running}
              onChange={(event) => setVisionConsent(event.target.checked)}
            />
            Also send up to four character references and two actual generated samples per run to my
            configured vision service. Qualitative appearance scores are not biometric identity
            checks. Vision API charges are separate from the GPU estimate.
          </label>
        </div>
      )}
      <fieldset disabled={running}>
        <legend>{mode === "action-discovery" ? "Model to probe" : "Models to compare"}</legend>
        <div className="we-models">
          {MODELS.map((model) => (
            <label key={model.id}>
              <input
                type={mode === "action-discovery" ? "radio" : "checkbox"}
                name="experiment-model"
                checked={selected.includes(model.id)}
                onChange={() =>
                  setSelected(
                    mode === "action-discovery"
                      ? [model.id]
                      : selected.includes(model.id)
                        ? selected.filter((id) => id !== model.id)
                        : [...selected, model.id],
                  )
                }
              />
              {model.name}
              <small>
                {model.status === "research" ? "requires configured adapter" : "adapter"}
              </small>
            </label>
          ))}
        </div>
      </fieldset>
      {mode === "compare" ? (
        <div className="we-trajectory">
          <label className="we-upload">
            Import control trajectory
            <input
              type="file"
              accept=".json,application/json"
              disabled={running}
              onChange={(event) => void importControls(event.target.files?.[0])}
            />
          </label>
          <span>
            {events.length
              ? `${events.length} events · ${(Math.max(...events.map((event) => event.timestampMs)) / 1000).toFixed(1)} s`
              : "No controls: stationary baseline"}{" "}
          </span>
          <button disabled={running || !events.length} onClick={() => setEvents([])}>
            Clear trajectory
          </button>
        </div>
      ) : mode === "character-consistency" ? (
        <p className="we-note">
          Samples, model/view settings, individual assessment evidence, and observed score averages
          are saved locally. With external vision off, captures and human ratings still work; no
          automatic scores are invented.
        </p>
      ) : mode === "return-to-view" ? (
        <p className="we-note">
          Move outward, pause, then apply the documented inverse movement for equal time. Compare
          the first and final image and inspect the recorded trajectory. This is visual return
          evidence, not a verified spatial map; clip adapters may apply actions at different times.
        </p>
      ) : (
        <p className="we-note">
          Runs a no-control baseline, then at most five documented native actions from the same
          initial seed. Measurements estimate visual motion; they do not infer semantic action
          meaning or prove causality. Unknown action vectors are never swept blindly.
        </p>
      )}
      <label className="we-consent">
        <input
          type="checkbox"
          checked={consent}
          disabled={running}
          onChange={(event) => setConsent(event.target.checked)}
        />
        I authorize these inference sessions. The selected compute receives this world prompt and
        reference image. Remote runs require a price quote; estimates are not billing guarantees.
      </label>
      <div className="we-actions">
        {running ? (
          <button
            className="we-primary"
            onClick={() => controller.current?.abort(new Error("Experiment cancelled."))}
          >
            <Square size={15} />
            Cancel and clean up
          </button>
        ) : (
          <button
            className="we-primary"
            disabled={
              !project ||
              !selected.length ||
              !consent ||
              (mode === "character-consistency" && (!character || !characterViews.length)) ||
              (mode === "action-discovery" && !getModel(selected[0] ?? "").nativeActions.length)
            }
            onClick={() => void start()}
          >
            <Play size={15} />
            Run{" "}
            {mode === "compare"
              ? "comparison"
              : mode === "return-to-view"
                ? "return experiment"
                : mode === "character-consistency"
                  ? "character experiments"
                  : "action discovery"}
          </button>
        )}
        <span>
          {project?.providerId === "local"
            ? "Local GPU · no cloud allocation"
            : "Configured provider · one worker at a time"}
        </span>
      </div>
      {progress && (
        <div className="we-progress" role="status">
          <strong>
            {progress.index}/{progress.total} · {progress.target}
          </strong>
          <span>{progress.phase}</span>
          <span>
            {progress.elapsedSeconds.toFixed(0)} s · {progress.frames} sampled frames · ~$
            {progress.estimatedCostUSD.toFixed(4)}
          </span>
        </div>
      )}
      {error && (
        <p className="we-error" role="alert">
          {error}
        </p>
      )}
      {!!all.length && (
        <div className="we-actions">
          <button
            onClick={() => {
              for (const video of resultGrid.current?.querySelectorAll("video") ?? []) {
                video.currentTime = 0;
                void video
                  .play()
                  .catch(() => setError("Click a video to enable browser playback."));
              }
            }}
          >
            Play results together
          </button>
          <button
            onClick={() => {
              for (const video of resultGrid.current?.querySelectorAll("video") ?? [])
                video.pause();
            }}
          >
            Pause all
          </button>
          <label>
            Compare at second{" "}
            <input
              aria-label="Comparison playback time"
              type="number"
              min="0"
              max="300"
              defaultValue="0"
              onChange={(event) => {
                const time = Number(event.target.value);
                if (Number.isFinite(time) && time >= 0)
                  for (const video of resultGrid.current?.querySelectorAll("video") ?? [])
                    video.currentTime = Math.min(
                      time,
                      Number.isFinite(video.duration) ? video.duration : time,
                    );
              }}
            />
          </label>
        </div>
      )}
      <div className="we-results" ref={resultGrid}>
        {all.map((result) => (
          <ResultCard key={result.id} result={result} onSaved={onSaved} />
        ))}
      </div>
      {!all.length && (
        <p className="we-empty">
          No experiment results yet. Completed and failed runs appear here with their actual
          evidence.
        </p>
      )}
    </section>
  );
}
