import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowLeft,
  Camera,
  Check,
  Circle,
  Command,
  Download,
  Expand,
  Keyboard,
  Mic,
  Pause,
  Play,
  Save,
  Settings2,
  Sparkles,
  Square,
  Volume2,
  VolumeX,
  X,
} from "lucide-react";
import type { ActionBinding, ControlEvent, ModelCapabilities, WorldProject } from "./core/types";
import { newId } from "./core/types";
import { getSettings, worldStore } from "./core/storage";
import {
  bindingSupport,
  exportTrajectory,
  parseTrajectory,
  replayTrajectory,
  suggestBindings,
} from "./core/controls";
import { getModel } from "./core/catalog";
import { createWorldApi } from "./core/api";
import { drawPreview, type PreviewView } from "./player/preview";
import { canUseContinuousControls, SessionActionQueue } from "./player/actions";
import { VoiceInput, type SpeechWindow } from "./player/voice";
import { ExplorationControls } from "./player/ExplorationControls";
import { explorationProgress } from "./player/exploration";
import { gamepadMotion } from "./player/gamepad";
import { IntelligencePanel, interpretCommand, actionSupported } from "./intelligence";
import {
  connectVideo,
  type LiveTransport,
  type VideoConnection,
  type StreamMetrics,
} from "./player/transport";
import { selectRetainedWorker, type ReusableWorker, type WorkerSession } from "./player/workers";
import "./player.css";

export interface PlayerProps {
  project: WorldProject;
  mode: "preview" | "live";
  serverUrl: string;
  onExit: () => void;
  onSaved?: () => void;
}
const formatTime = (seconds: number) =>
  `${Math.floor(seconds / 60)
    .toString()
    .padStart(2, "0")}:${Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0")}`;
const isTyping = (target: EventTarget | null) =>
  target instanceof HTMLElement &&
  (["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(target.tagName) || target.isContentEditable);
const errorText = (error: unknown) => (error instanceof Error ? error.message : String(error));
function saveDownload(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function captureCanvas(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) =>
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("The frame could not be captured."))),
      "image/png",
    ),
  );
}
function renderFrame(
  ctx: CanvasRenderingContext2D,
  source: HTMLVideoElement | ImageBitmap,
  width: number,
  height: number,
) {
  const sourceWidth = source instanceof HTMLVideoElement ? source.videoWidth : source.width;
  const sourceHeight = source instanceof HTMLVideoElement ? source.videoHeight : source.height;
  if (!sourceWidth || !sourceHeight) return;
  const scale = Math.min(width / sourceWidth, height / sourceHeight);
  ctx.fillStyle = "#070b0c";
  ctx.fillRect(0, 0, width, height);
  ctx.drawImage(
    source,
    (width - sourceWidth * scale) / 2,
    (height - sourceHeight * scale) / 2,
    sourceWidth * scale,
    sourceHeight * scale,
  );
}
function wait(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(done, ms);
    function done() {
      signal.removeEventListener("abort", abort);
      resolve();
    }
    function abort() {
      clearTimeout(timer);
      reject(new DOMException("Session cancelled", "AbortError"));
    }
    if (signal.aborted) abort();
    else signal.addEventListener("abort", abort, { once: true });
  });
}

export default function Player({ project, mode, serverUrl, onExit, onSaved }: PlayerProps) {
  const preview = mode === "preview";
  const model = getModel(project.modelId);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const screenRef = useRef<HTMLDivElement>(null);
  const started = useRef(0);
  const events = useRef<ControlEvent[]>([]);
  const view = useRef<PreviewView>({ travel: 0, yaw: 0, pitch: 0, hue: 177, rain: false });
  const pressed = useRef(new Map<string, ActionBinding>());
  const gamepadHeld = useRef(new Map<number, ActionBinding>());
  const analog = useRef({ active: false, sentAt: 0 });
  const refreshingHeld = useRef(false);
  const pausedRef = useRef(false);
  const latestFrame = useRef<ImageBitmap | null>(null);
  const frameCount = useRef(0);
  const videoFrameCount = useRef(0);
  const hasFrame = useRef(false);
  const session = useRef<{
    id: string;
    workerId: string;
    managed?: boolean;
    sessionDeleted?: boolean;
    workerDeleted?: boolean;
  } | null>(null);
  const trajectoryController = useRef<AbortController | null>(null);
  const lastActivity = useRef(0);
  const effectiveSeed = useRef(project.settings.seed);
  const actionQueue = useRef(new SessionActionQueue());
  const mouseDelta = useRef({ x: 0, y: 0, sentAt: 0 });
  const settings = useMemo(() => getSettings(), []);
  const cleanupStream = useRef<VideoConnection | null>(null);
  const lifecycle = useRef<AbortController | null>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const recordStream = useRef<MediaStream | null>(null);
  const speech = useRef<VoiceInput | null>(null);
  const voiceRequest = useRef<AbortController | null>(null);
  const mounted = useRef(true);
  const [capabilities, setCapabilities] = useState<ModelCapabilities | undefined>(undefined);
  const [nativeActions, setNativeActions] = useState<string[]>([]);
  const [status, setStatus] = useState(
    preview ? "Interface preview · no GPU" : "Connecting to compute…",
  );
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [paused, setPaused] = useState(false);
  const [revisions, setRevisions] = useState<{
    queued?: number;
    applied?: number;
    generating?: number | null;
    promptTruncated?: boolean;
  }>({});
  const [clipAction, setClipAction] = useState("stop");
  const [elapsed, setElapsed] = useState(0);
  const [fps, setFps] = useState(0);
  const [receivedFrames, setReceivedFrames] = useState(0);
  const [connected, setConnected] = useState(false);
  const [streamMetrics, setStreamMetrics] = useState<StreamMetrics>({});
  const [generationMetrics, setGenerationMetrics] = useState<{
    generatedFPS?: number;
    generationSeconds?: number;
    modelLoadSeconds?: number;
    modelLoadCount?: number;
  }>({});
  const [recording, setRecording] = useState(false);
  const [muted, setMuted] = useState(true);
  const [recordingBusy, setRecordingBusy] = useState(false);
  const [panel, setPanel] = useState<"controls" | "session" | "director" | null>(null);
  const [prompt, setPrompt] = useState("");
  const [activePrompt, setActivePrompt] = useState(project.prompt);
  const [recentEvents, setRecentEvents] = useState<ControlEvent[]>([]);
  const [voiceConsent, setVoiceConsent] = useState(false);
  const [voiceAutoSend, setVoiceAutoSend] = useState(true);
  const [voiceContinuous, setVoiceContinuous] = useState(false);
  const [voiceSceneAware, setVoiceSceneAware] = useState(false);
  const [voiceDialog, setVoiceDialog] = useState(false);
  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [bindings, setBindings] = useState<ActionBinding[]>([]);
  const [bindingKey, setBindingKey] = useState("KeyF");
  const [bindingText, setBindingText] = useState("Make the sky turn to sunset");
  const [bindingType, setBindingType] = useState<"native" | "prompt" | "semantic">("prompt");
  const [gamepadButton, setGamepadButton] = useState("");
  const [exiting, setExiting] = useState(false);
  const [busy, setBusy] = useState(false);
  const [hourlyCost, setHourlyCost] = useState<number | null>(preview ? 0 : null);
  const [workerManaged, setWorkerManaged] = useState(false);
  const [replaying, setReplaying] = useState(false);
  const continuousChunks = capabilities?.runtime.interactionMode === "continuous-chunks";
  const promptAdapted = capabilities?.runtime.controlMode === "prompt-adapted";
  const offlineClip = capabilities?.runtime.interactionMode === "offline-clip";
  const canPrompt =
    preview ||
    capabilities?.control.promptDuringRollout === true ||
    capabilities?.control.promptSwitching === true ||
    capabilities?.control.semanticActions === true;

  const api = useMemo<LiveTransport>(() => {
    const client = createWorldApi(serverUrl);
    return {
      request: client.request,
      frame: (path, signal) =>
        client.request<Blob | null>(
          path,
          { signal: AbortSignal.any([signal, AbortSignal.timeout(15000)]), cache: "no-store" },
          true,
        ),
    };
  }, [serverUrl]);

  const report = useCallback((message: string) => {
    if (mounted.current) setNotice(message);
  }, []);
  const teardown = useCallback(async () => {
    lifecycle.current?.abort();
    trajectoryController.current?.abort();
    cleanupStream.current?.();
    cleanupStream.current = null;
    voiceRequest.current?.abort();
    speech.current?.stop();
    speech.current = null;
    if (document.pointerLockElement) document.exitPointerLock();
    const current = session.current;
    if (current) {
      const errors: string[] = [];
      try {
        if (!current.sessionDeleted) {
          await api.request(`/sessions/${current.id}`, { method: "DELETE", keepalive: true });
          current.sessionDeleted = true;
        }
      } catch (error) {
        errors.push(errorText(error));
      }
      try {
        if (!current.workerDeleted && !settings.retainWorker) {
          await api.request(`/workers/${current.workerId}`, { method: "DELETE", keepalive: true });
          current.workerDeleted = true;
        }
      } catch (error) {
        errors.push(errorText(error));
      }
      if (current.managed && current.workerDeleted) errors.length = 0;
      if (!errors.length) session.current = null;
      if (errors.length)
        throw new Error(
          `Cleanup could not be confirmed: ${errors.join("; ")}. Check your provider console for running workers.`,
        );
    }
  }, [api, settings.retainWorker]);

  useEffect(() => {
    mounted.current = true;
    started.current = performance.now();
    lastActivity.current = performance.now();
    const controller = new AbortController();
    lifecycle.current = controller;
    let ownedWorker: string | undefined;
    let reusedWorker = false;
    async function start() {
      if (preview) return;
      try {
        setStatus("Checking local references…");
        if (
          (model.capabilities.input.text && !project.prompt.trim()) ||
          project.prompt.length > 8000
        )
          throw new Error("World prompts must contain 1–8000 characters.");
        if (
          project.settings.seed !== undefined &&
          (!Number.isSafeInteger(project.settings.seed) ||
            project.settings.seed < 0 ||
            project.settings.seed > 4294967295)
        )
          throw new Error("Seed must be an unsigned 32-bit integer.");
        const inputs: { images?: string[]; video?: string } = {};
        let total = 0;
        for (const id of project.assetIds) {
          const blob = await worldStore.getBlob(id);
          if (!blob)
            throw new Error(
              "A reference asset is missing from local storage. Remove it or upload it again before starting a session.",
            );
          if (
            !["image/jpeg", "image/png", "image/webp", "video/mp4", "video/webm"].includes(
              blob.type,
            )
          )
            throw new Error("Unsupported reference media type.");
          if (blob.type.startsWith("video/") && !model.capabilities.input.video)
            throw new Error("This model adapter does not support video references.");
          total += blob.size;
          if (total > 4 * 1024 * 1024)
            throw new Error(
              "Reference media exceeds the 4 MB session limit. Use smaller references.",
            );
          const data = await new Promise<string>((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () =>
              typeof reader.result === "string"
                ? resolve(reader.result)
                : reject(new Error("Invalid media data"));
            reader.onerror = () => reject(new Error("Could not read reference media"));
            reader.readAsDataURL(blob);
          });
          if (blob.type.startsWith("image/")) (inputs.images ??= []).push(data);
          else if (blob.type.startsWith("video/")) inputs.video = data;
        }
        if (model.capabilities.input.requiredImage && !inputs.images?.length)
          throw new Error("This model requires a starting image before compute can start.");
        if (
          model.capabilities.runtime.qualityOptions &&
          !model.capabilities.runtime.qualityOptions.includes(project.settings.performance)
        )
          throw new Error("This model does not support the selected performance preset.");
        controller.signal.throwIfAborted();
        setStatus(
          `Connecting ${project.providerId === "runpod" ? "RunPod" : project.providerId} worker…`,
        );
        let available: ReusableWorker | undefined;
        if (settings.retainWorker) {
          setStatus("Looking for your retained worker…");
          const [workers, sessions] = await Promise.all([
            api.request<{ workers: ReusableWorker[] }>("/workers", { signal: controller.signal }),
            api.request<{ sessions: WorkerSession[] }>("/sessions", { signal: controller.signal }),
          ]);
          available = selectRetainedWorker(
            workers.workers,
            sessions.sessions,
            project.providerId,
            project.modelId,
          );
          reusedWorker = available !== undefined;
        }
        controller.signal.throwIfAborted();
        let worker =
          available ??
          (await api.request<ReusableWorker>("/workers", {
            method: "POST",
            body: JSON.stringify({ provider: project.providerId, modelId: project.modelId }),
          }));
        ownedWorker = worker.id;
        if (controller.signal.aborted) {
          if (!reusedWorker) await api.request(`/workers/${worker.id}`, { method: "DELETE" });
          return;
        }
        if (reusedWorker) setStatus("Reusing your retained GPU worker…");
        setWorkerManaged(worker.managed === true);
        if (typeof worker.estimatedHourlyCost === "number")
          setHourlyCost(worker.estimatedHourlyCost);
        const bootDeadline = performance.now() + 15 * 60 * 1000;
        while (worker.status !== "ready") {
          if (["error", "unknown", "stopped", "destroyed", "detached"].includes(worker.status))
            throw new Error(
              `Worker is ${worker.status}. Review worker recovery in Settings before trying again.`,
            );
          if (performance.now() > bootDeadline)
            throw new Error(
              "Worker startup exceeded 15 minutes. Check the provider console and worker logs.",
            );
          setStatus(`Worker ${worker.status} · waiting for gateway readiness…`);
          await wait(3000, controller.signal);
          worker = await api.request<typeof worker>(`/workers/${worker.id}`, {
            signal: controller.signal,
          });
          if (typeof worker.estimatedHourlyCost === "number")
            setHourlyCost(worker.estimatedHourlyCost);
        }
        setStatus("Preparing model session…");
        const created = await api.request<{
          id: string;
          seed?: number;
          capabilities?: ModelCapabilities & { nativeActions?: string[] };
          nativeActions?: string[];
        }>("/sessions", {
          method: "POST",
          body: JSON.stringify({
            workerId: worker.id,
            modelId: project.modelId,
            prompt: model.capabilities.input.text ? project.prompt : "",
            seed: project.settings.seed,
            quality: project.settings.performance,
            resolution: project.settings.resolution,
            inputs,
          }),
        });
        if (controller.signal.aborted) {
          await api.request(`/sessions/${created.id}`, { method: "DELETE" });
          if (!reusedWorker) await api.request(`/workers/${worker.id}`, { method: "DELETE" });
          return;
        }
        session.current = { id: created.id, workerId: worker.id, managed: worker.managed === true };
        lastActivity.current = performance.now();
        effectiveSeed.current = created.seed ?? project.settings.seed;
        setCapabilities(created.capabilities);
        setConnected(true);
        setNativeActions(
          created.nativeActions ??
            created.capabilities?.nativeActions ??
            (created.capabilities?.control.wasd ? ["forward", "backward", "left", "right"] : []),
        );
        setStatus("Connecting video stream…");
        if (videoRef.current)
          cleanupStream.current = await connectVideo(
            api,
            created.id,
            videoRef.current,
            controller.signal,
            (state) => {
              if (!controller.signal.aborted) setStatus(state);
            },
            (frame) => {
              latestFrame.current?.close();
              latestFrame.current = frame;
              frameCount.current++;
              hasFrame.current = true;
            },
            (metrics) => {
              if (!controller.signal.aborted)
                setStreamMetrics((current) => ({ ...current, ...metrics }));
            },
            { audio: created.capabilities?.output.audio === true },
          );
        while (!controller.signal.aborted) {
          await wait(4000, controller.signal);
          const state = await api.request<{
            status: string;
            stage?: string;
            error?: string;
            generatedFPS?: number;
            generationSeconds?: number;
            modelLoadSeconds?: number;
            modelLoadCount?: number;
            queuedRevision?: number;
            appliedRevision?: number;
            generatingRevision?: number | null;
            promptTruncated?: boolean;
          }>(`/sessions/${created.id}`, { signal: controller.signal });
          setRevisions((current) => ({
            queued:
              state.queuedRevision === undefined
                ? current.queued
                : Math.max(current.queued ?? 0, state.queuedRevision),
            applied: state.appliedRevision,
            generating: state.generatingRevision,
            promptTruncated: state.promptTruncated === true,
          }));
          setGenerationMetrics({
            generatedFPS: state.generatedFPS,
            generationSeconds: state.generationSeconds,
            modelLoadSeconds: state.modelLoadSeconds,
            modelLoadCount: state.modelLoadCount,
          });
          if (state.status === "paused" && !pausedRef.current) {
            pausedRef.current = true;
            setPaused(true);
            speech.current?.stop();
            voiceRequest.current?.abort();
            pressed.current.clear();
            gamepadHeld.current.clear();
            setStatus(
              created.capabilities?.runtime.interactionMode === "offline-clip"
                ? "Clip complete · resume to generate the next clip"
                : "Generation paused",
            );
          }
          if (state.error || state.status === "error") {
            setError(
              state.error ??
                "Model generation failed. Check the worker configuration and GPU memory.",
            );
            setStatus("Generation failed");
            break;
          }
          if (!hasFrame.current && state.stage) setStatus(state.stage);
          else if (!hasFrame.current) setStatus(`Model ${state.status} · waiting for first frame…`);
        }
      } catch (error) {
        // A retained worker can be claimed by another tab between discovery and POST.
        // Without our own returned session ID, its ledger must remain untouched.
        if (ownedWorker && !session.current && !reusedWorker) {
          try {
            const records = await api.request<{
              sessions: { id: string; workerId: string; status: string }[];
            }>("/sessions");
            for (const record of records.sessions.filter(
              (item) => item.workerId === ownedWorker && item.status !== "stopped",
            ))
              await api.request(`/sessions/${record.id}`, { method: "DELETE" });
            await api.request(`/workers/${ownedWorker}`, { method: "DELETE" });
          } catch {
            if (!controller.signal.aborted)
              setNotice(
                "Worker cleanup could not be confirmed. Check your provider console and worker recovery in Settings.",
              );
          }
        }
        if (!controller.signal.aborted) {
          setError(
            reusedWorker && !session.current
              ? `${errorText(error)} The retained worker was left unchanged. It may be in use by another tab; inspect its sessions in Settings before trying again.`
              : errorText(error),
          );
          setStatus(session.current ? "Session connection lost" : "Session could not start");
        }
      }
    }
    void start();
    return () => {
      mounted.current = false;
      controller.abort();
      void teardown().catch(() => {
        /* Session may already be closed. */
      });
      if (recorder.current?.state !== "inactive") recorder.current?.stop();
      recordStream.current?.getTracks().forEach((track) => track.stop());
      latestFrame.current?.close();
      latestFrame.current = null;
    };
  }, [api, model, preview, project, settings.retainWorker, teardown]);

  useEffect(() => {
    const basic: ActionBinding[] =
      preview || capabilities?.control.wasd
        ? [
            {
              id: "forward",
              key: "KeyW",
              label: "Move forward",
              type: "native",
              action: "forward",
              gamepadButton: 12,
            },
            {
              id: "backward",
              key: "KeyS",
              label: "Move backward",
              type: "native",
              action: "backward",
              gamepadButton: 13,
            },
            {
              id: "left",
              key: "KeyA",
              label: "Move left",
              type: "native",
              action: "left",
              gamepadButton: 14,
            },
            {
              id: "right",
              key: "KeyD",
              label: "Move right",
              type: "native",
              action: "right",
              gamepadButton: 15,
            },
          ]
        : [];
    try {
      const saved = localStorage.getItem(
        `worlds-bindings:${preview ? "preview" : project.modelId}`,
      );
      if (saved) {
        const list: unknown = JSON.parse(saved);
        if (Array.isArray(list)) {
          queueMicrotask(() =>
            setBindings(
              list.filter(
                (item: ActionBinding) =>
                  item &&
                  typeof item.key === "string" &&
                  ["native", "prompt", "semantic"].includes(item.type),
              ),
            ),
          );
          return;
        }
      }
    } catch {
      /* Local bindings are optional. */
    }
    queueMicrotask(() => setBindings(basic));
  }, [capabilities, preview, project.modelId]);

  const submit = useCallback(
    async (input: Omit<ControlEvent, "id" | "timestampMs">) => {
      lastActivity.current = performance.now();
      const timestampMs = Math.round(lastActivity.current - started.current);
      if (
        pausedRef.current &&
        input.type !== "resume" &&
        !(
          capabilities?.runtime.interactionMode === "offline-clip" &&
          (input.type === "prompt" || (input.type === "native" && input.values?.planned === true))
        ) &&
        !(input.type === "native" && input.values?.pressed === false)
      )
        return;
      if (!preview) {
        if (!session.current) throw new Error("The model session is not connected.");
        if (
          input.type === "prompt" &&
          !(capabilities?.control.promptDuringRollout || capabilities?.control.promptSwitching)
        )
          throw new Error("This worker does not advertise live prompt support.");
        if (input.type === "semantic" && !capabilities?.control.semanticActions)
          throw new Error("This worker does not advertise semantic actions.");
        if (
          input.type === "native" &&
          input.action !== "look" &&
          !nativeActions.includes(input.action ?? "")
        )
          throw new Error("This action is not advertised by the connected worker.");
        if (
          input.type === "native" &&
          !(
            capabilities?.control.wasd ||
            capabilities?.control.discreteActions ||
            capabilities?.control.continuousActions ||
            capabilities?.control.mouseLook
          )
        )
          throw new Error("Native actions are unavailable on this worker.");
        const sessionId = session.current.id;
        const sessionSignal = lifecycle.current?.signal;
        if (!sessionSignal) throw new Error("The session is no longer active.");
        const acknowledgment: unknown = await actionQueue.current.enqueue(
          () =>
            cleanupStream.current?.sendAction(input) ??
            api.request(`/sessions/${sessionId}/actions`, {
              method: "POST",
              body: JSON.stringify(input),
              signal: AbortSignal.any([sessionSignal, AbortSignal.timeout(5000)]),
            }),
          sessionSignal,
          input.type === "native" &&
            input.action !== "exploration" &&
            input.values?.pressed !== false
            ? 1000
            : undefined,
        );
        sessionSignal.throwIfAborted();
        if (
          acknowledgment &&
          typeof acknowledgment === "object" &&
          "revision" in acknowledgment &&
          typeof acknowledgment.revision === "number" &&
          Number.isSafeInteger(acknowledgment.revision)
        ) {
          const revision = acknowledgment.revision;
          setRevisions((current) => ({
            ...current,
            queued: Math.max(current.queued ?? 0, revision),
          }));
        }
      }
      const event: ControlEvent = { ...input, id: newId(), timestampMs };
      if (input.type === "pause" || input.type === "resume") {
        pausedRef.current = input.type === "pause";
        setPaused(input.type === "pause");
      }
      events.current.push(event);
      if (events.current.length > 50000) events.current.shift();
      if (input.prompt) {
        setActivePrompt(input.prompt);
        if (preview) {
          view.current.rain = /rain|storm|snow/i.test(input.prompt);
          view.current.hue = /sunset|warm|fire/i.test(input.prompt)
            ? 25
            : /alien|purple|dream/i.test(input.prompt)
              ? 270
              : /ice|snow|blue/i.test(input.prompt)
                ? 211
                : 177;
          report("Preview palette updated. These are procedural effects, not AI responses.");
        }
      }
      if (input.type !== "native")
        setRecentEvents(events.current.filter((item) => item.type !== "native").slice(-12));
    },
    [api, capabilities, nativeActions, preview, report],
  );

  const releaseAnalog = useCallback(() => {
    if (!analog.current.active) return;
    analog.current.active = false;
    void submit({
      type: "native",
      action: "analog",
      values: { forward: 0, right: 0, up: 0, yaw: 0, pitch: 0, pressed: false },
    }).catch(() => {
      /* Session teardown may have already stopped input. */
    });
  }, [submit]);

  const releaseInputs = useCallback(() => {
    releaseAnalog();
    for (const binding of [...pressed.current.values(), ...gamepadHeld.current.values()])
      if (binding.type === "native")
        void submit({ type: "native", action: binding.action, values: { pressed: false } }).catch(
          () => {
            /* Session may already be closed. */
          },
        );
    pressed.current.clear();
    gamepadHeld.current.clear();
  }, [submit, releaseAnalog]);

  useEffect(() => {
    let raf = 0,
      last = performance.now(),
      fpsTime = last,
      lastFrames = frameCount.current;
    const gamepadPressed = new Map<number, ActionBinding>();
    gamepadHeld.current = gamepadPressed;
    const releaseGamepad = () => {
      releaseAnalog();
      for (const binding of gamepadPressed.values())
        if (binding.type === "native")
          void submit({ type: "native", action: binding.action, values: { pressed: false } }).catch(
            () => {
              /* The session may be disconnected. */
            },
          );
      gamepadPressed.clear();
    };
    function run(now: number) {
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext("2d");
      const dt = Math.min((now - last) / 1000, 0.1);
      last = now;
      if (canvas && ctx) {
        const width = Math.round(canvas.clientWidth * Math.min(devicePixelRatio, 1.5));
        const height = Math.round(canvas.clientHeight * Math.min(devicePixelRatio, 1.5));
        if (canvas.width !== width || canvas.height !== height) {
          canvas.width = Math.max(width, 1);
          canvas.height = Math.max(height, 1);
        }
        if (preview && !pausedRef.current) {
          for (const binding of [...pressed.current.values(), ...gamepadPressed.values()]) {
            if (binding.action === "forward") view.current.travel += dt * 90;
            if (binding.action === "backward") view.current.travel -= dt * 90;
            if (binding.action === "left") view.current.yaw -= dt * 0.4;
            if (binding.action === "right") view.current.yaw += dt * 0.4;
          }
          drawPreview(ctx, canvas.width, canvas.height, now, view.current);
          frameCount.current++;
          hasFrame.current = true;
        } else if (!preview) {
          const video = videoRef.current;
          if (video && video.readyState >= 2 && video.srcObject) {
            renderFrame(ctx, video, canvas.width, canvas.height);
            hasFrame.current = true;
            const decoded = video.getVideoPlaybackQuality?.().totalVideoFrames ?? 0;
            frameCount.current += Math.max(0, decoded - videoFrameCount.current);
            videoFrameCount.current = decoded;
          } else if (latestFrame.current)
            renderFrame(ctx, latestFrame.current, canvas.width, canvas.height);
        }
      }
      if (
        canUseContinuousControls({
          visible: document.visibilityState === "visible",
          focused: document.hasFocus(),
          paused: pausedRef.current,
          menuOpen: panel !== null,
          typing: isTyping(document.activeElement),
        })
      ) {
        const gamepad = navigator.getGamepads?.().find((pad) => pad?.connected);
        if (!gamepad) releaseGamepad();
        if (
          gamepad &&
          capabilities?.control.gamepad &&
          nativeActions.includes("analog") &&
          now - analog.current.sentAt >= 100
        ) {
          const motion = gamepadMotion(gamepad.axes);
          const active = Object.values(motion).some((value) => value !== 0);
          if (active || analog.current.active) {
            analog.current = { active, sentAt: now };
            void submit({
              type: "native",
              action: "analog",
              values: { ...motion, pressed: active },
            }).catch((error: unknown) => report(errorText(error)));
          }
        }
        if (gamepad)
          for (const binding of bindings) {
            const index = binding.gamepadButton;
            if (index === undefined) continue;
            const down = !!gamepad.buttons[index]?.pressed;
            if (down && !gamepadPressed.has(index)) {
              gamepadPressed.set(index, binding);
              void submit({
                type: binding.type,
                action: binding.action,
                prompt: binding.prompt,
                values: binding.type === "native" ? { pressed: true } : undefined,
              }).catch((error: unknown) => report(errorText(error)));
            }
            if (!down && gamepadPressed.has(index)) {
              gamepadPressed.delete(index);
              if (binding.type === "native")
                void submit({
                  type: "native",
                  action: binding.action,
                  values: { pressed: false },
                }).catch(() => {
                  /* Session may already be closed. */
                });
            }
          }
      } else releaseGamepad();
      if (now - fpsTime >= 1000) {
        setReceivedFrames(frameCount.current);
        setFps(Math.round(((frameCount.current - lastFrames) * 1000) / (now - fpsTime)));
        lastFrames = frameCount.current;
        fpsTime = now;
        setElapsed((now - started.current) / 1000);
      }
      raf = requestAnimationFrame(run);
    }
    raf = requestAnimationFrame(run);
    return () => {
      cancelAnimationFrame(raf);
      releaseGamepad();
    };
  }, [preview, bindings, panel, submit, report, capabilities, nativeActions, releaseAnalog]);

  useEffect(() => {
    if (preview) return;
    const timer = setInterval(() => {
      if (
        refreshingHeld.current ||
        !canUseContinuousControls({
          visible: !document.hidden,
          focused: document.hasFocus(),
          paused: pausedRef.current,
          menuOpen: panel !== null,
          typing: isTyping(document.activeElement),
        })
      )
        return;
      const held = [...pressed.current.values(), ...gamepadHeld.current.values()].filter(
        (item) => item.type === "native",
      );
      const renew = capabilities?.control.continuousActions ? held : held.slice(-1);
      if (!renew.length) return;
      refreshingHeld.current = true;
      void Promise.all(
        renew.map((binding) =>
          submit({ type: "native", action: binding.action, values: { pressed: true, held: true } }),
        ),
      )
        .catch((error: unknown) => report(errorText(error)))
        .finally(() => {
          refreshingHeld.current = false;
        });
    }, 250);
    return () => clearInterval(timer);
  }, [panel, preview, report, submit, capabilities]);

  useEffect(() => {
    function keydown(event: KeyboardEvent) {
      if (isTyping(event.target)) return;
      if (event.code === "Slash") {
        event.preventDefault();
        releaseInputs();
        inputRef.current?.focus();
        return;
      }
      if (event.code === "Escape") {
        setPanel(null);
        releaseInputs();
        return;
      }
      if (
        panel ||
        event.repeat ||
        event.ctrlKey ||
        event.altKey ||
        event.metaKey ||
        pausedRef.current
      )
        return;
      const binding = bindings.find((item) => item.key === event.code);
      if (!binding) return;
      event.preventDefault();
      pressed.current.set(event.code, binding);
      void submit({
        type: binding.type,
        action: binding.action,
        prompt: binding.prompt,
        values: binding.type === "native" ? { pressed: true } : undefined,
      }).catch((error: unknown) => report(errorText(error)));
    }
    function keyup(event: KeyboardEvent) {
      const binding = pressed.current.get(event.code);
      pressed.current.delete(event.code);
      if (binding?.type === "native")
        void submit({ type: "native", action: binding.action, values: { pressed: false } }).catch(
          () => {
            /* Session may already be closed. */
          },
        );
    }
    const hidden = () => {
      if (document.hidden) loseFocus();
    };
    const loseFocus = () => {
      releaseInputs();
      voiceRequest.current?.abort();
      speech.current?.stop();
      setListening(false);
    };
    window.addEventListener("keydown", keydown);
    window.addEventListener("keyup", keyup);
    window.addEventListener("blur", loseFocus);
    document.addEventListener("visibilitychange", hidden);
    return () => {
      releaseInputs();
      window.removeEventListener("keydown", keydown);
      window.removeEventListener("keyup", keyup);
      window.removeEventListener("blur", loseFocus);
      document.removeEventListener("visibilitychange", hidden);
    };
  }, [bindings, panel, submit, releaseInputs, report]);

  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(""), 6500);
    return () => clearTimeout(timer);
  }, [notice]);

  useEffect(() => {
    if (preview || !connected) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    async function heartbeat() {
      const current = session.current;
      if (!current || stopped || lifecycle.current?.signal.aborted) return;
      try {
        await api.request(`/sessions/${current.id}/heartbeat`, {
          method: "POST",
          body: JSON.stringify({
            active:
              document.visibilityState === "visible" &&
              document.hasFocus() &&
              performance.now() - lastActivity.current < settings.idleTimeoutMinutes * 60000,
          }),
          signal: AbortSignal.any([
            lifecycle.current?.signal ?? AbortSignal.timeout(8000),
            AbortSignal.timeout(8000),
          ]),
        });
      } catch (error) {
        if (!stopped && !lifecycle.current?.signal.aborted)
          report(
            `Session heartbeat could not be renewed: ${errorText(error)}. The server may stop this session to limit GPU costs.`,
          );
      }
      if (!stopped) timer = setTimeout(() => void heartbeat(), 20000);
    }
    void heartbeat();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [api, connected, preview, report, settings.idleTimeoutMinutes]);

  useEffect(() => {
    if (preview) return;
    const timer = setInterval(() => {
      if (
        !session.current ||
        performance.now() - lastActivity.current < settings.idleTimeoutMinutes * 60000
      )
        return;
      lastActivity.current = performance.now();
      void teardown()
        .then(() => {
          setConnected(false);
          setStatus("Session ended after inactivity");
          setError(
            settings.retainWorker
              ? "The session ended after your configured idle timeout. Worker retention is enabled; GPU billing may continue."
              : "The session ended after your configured idle timeout. Managed worker teardown was requested.",
          );
        })
        .catch((error: unknown) => setError(errorText(error)));
    }, 10000);
    return () => clearInterval(timer);
  }, [preview, settings.idleTimeoutMinutes, settings.retainWorker, teardown]);

  async function sendPrompt(text: string) {
    if (
      !text.trim() ||
      !canPrompt ||
      (pausedRef.current && !offlineClip) ||
      exiting ||
      lifecycle.current?.signal.aborted
    )
      return;
    try {
      await submit({
        type:
          !preview &&
          capabilities?.control.semanticActions &&
          !(capabilities.control.promptDuringRollout || capabilities.control.promptSwitching)
            ? "semantic"
            : "prompt",
        prompt: text.trim(),
      });
      setPrompt("");
      inputRef.current?.blur();
    } catch (error) {
      report(errorText(error));
    }
  }
  async function togglePause() {
    try {
      voiceRequest.current?.abort();
      speech.current?.stop();
      releaseInputs();
      if (paused && offlineClip) {
        if (prompt.trim()) {
          await submit({ type: "prompt", prompt: prompt.trim() });
          setPrompt("");
        }
        await submit({ type: "native", action: clipAction, values: { planned: true } });
      }
      await submit({ type: paused ? "resume" : "pause" });
    } catch (error) {
      report(errorText(error));
    }
  }
  async function saveScene() {
    if (!canvasRef.current || !hasFrame.current)
      return report("Wait for the first frame before saving.");
    setBusy(true);
    try {
      const blob = await captureCanvas(canvasRef.current);
      const asset = await worldStore.saveAsset(blob, `${project.name} checkpoint.png`, "image");
      await worldStore.put("scenes", {
        id: newId(),
        name: `${project.name} · ${formatTime(elapsed)}`,
        projectId: project.id,
        modelId: project.modelId,
        createdAt: Date.now(),
        prompt: activePrompt,
        resumeKind: "visual",
        thumbnailAssetId: asset.id,
        assetIds: [asset.id],
        events: [...events.current],
        seed: effectiveSeed.current,
      });
      report(
        "Visual checkpoint saved locally. It captures the frame and trajectory; exact model resume is not available.",
      );
      onSaved?.();
    } catch (error) {
      report(`Save failed: ${errorText(error)}`);
    } finally {
      setBusy(false);
    }
  }
  async function screenshot() {
    if (!canvasRef.current || !hasFrame.current)
      return report("Wait for the first frame before capturing.");
    try {
      saveDownload(
        await captureCanvas(canvasRef.current),
        `${project.name.replace(/[^\w-]/g, "-")}-${preview ? "preview" : "frame"}.png`,
      );
      report("Screenshot downloaded.");
    } catch (error) {
      report(errorText(error));
    }
  }
  function toggleRecording() {
    if (recorder.current && recorder.current.state !== "inactive") {
      recorder.current.stop();
      return;
    }
    if (!canvasRef.current || !hasFrame.current)
      return report("Wait for the first frame before recording.");
    if (typeof MediaRecorder === "undefined" || !canvasRef.current.captureStream)
      return report("Video recording is not available in this browser.");
    try {
      const stream = canvasRef.current.captureStream(30);
      const incoming = videoRef.current?.srcObject;
      if (incoming instanceof MediaStream)
        for (const track of incoming.getAudioTracks()) stream.addTrack(track.clone());
      recordStream.current = stream;
      const mimeType = [
        "video/webm;codecs=vp9,opus",
        "video/webm;codecs=vp8,opus",
        "video/webm",
        "video/mp4",
      ].find((type) => MediaRecorder.isTypeSupported(type));
      const instance = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      recorder.current = instance;
      const chunks: Blob[] = [];
      let bytes = 0;
      const from = performance.now();
      const eventStart = events.current.length;
      instance.ondataavailable = (event) => {
        if (event.data.size) {
          chunks.push(event.data);
          bytes += event.data.size;
        }
        if (bytes > 256 * 1024 * 1024 && instance.state === "recording") {
          instance.stop();
          report("Recording reached the 256 MB memory limit and is being saved.");
        }
      };
      instance.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        recorder.current = null;
        if (mounted.current) {
          setRecording(false);
          setRecordingBusy(true);
        }
        const blob = new Blob(chunks, { type: instance.mimeType || "video/webm" });
        const save = async () => {
          const asset = await worldStore.saveAsset(
            blob,
            `${project.name} replay.${blob.type.includes("mp4") ? "mp4" : "webm"}`,
            "video",
          );
          await worldStore.put("replays", {
            id: newId(),
            name: `${project.name} · ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`,
            projectId: project.id,
            assetId: asset.id,
            createdAt: Date.now(),
            durationMs: performance.now() - from,
            modelId: project.modelId,
            events: events.current.slice(eventStart).map((event) => ({
              ...event,
              timestampMs: Math.max(0, event.timestampMs - (from - started.current)),
            })),
            previewOnly: preview,
          });
          onSaved?.();
          report("Replay saved to your local library.");
        };
        void save()
          .catch((error: unknown) => {
            report(
              `Local replay save failed: ${errorText(error)}. A download was started instead.`,
            );
            saveDownload(blob, "world-replay.webm");
          })
          .finally(() => {
            if (mounted.current) setRecordingBusy(false);
          });
      };
      instance.onerror = () => report("The browser stopped recording unexpectedly.");
      instance.start(1000);
      setRecording(true);
      report("Recording the world output. Microphone audio is not recorded.");
    } catch (error) {
      recordStream.current?.getTracks().forEach((track) => track.stop());
      report(errorText(error));
    }
  }
  const currentFrame = useCallback(async () => {
    if (!canvasRef.current || !hasFrame.current) return undefined;
    return captureCanvas(canvasRef.current);
  }, []);

  async function sendVoice(text: string) {
    if (!voiceSceneAware || preview) return sendPrompt(text);
    if (voiceRequest.current && !voiceRequest.current.signal.aborted) {
      setPrompt(text);
      report("Still interpreting the previous phrase. Your next command is in the command bar.");
      return;
    }
    const controller = new AbortController();
    voiceRequest.current = controller;
    try {
      const frame = await currentFrame();
      if (!frame || !capabilities)
        throw new Error("Wait for a generated frame before using scene-aware voice.");
      const proposal = await interpretCommand({
        serverUrl,
        modelId: project.modelId,
        capabilities,
        nativeActions,
        frame,
        prompt: events.current.filter((event) => event.prompt).at(-1)?.prompt ?? project.prompt,
        command: text,
        objective: project.game?.objective,
        recentEvents: events.current
          .slice(-20)
          .map((event) => event.prompt ?? event.action ?? event.type),
        signal: controller.signal,
      });
      controller.signal.throwIfAborted();
      if (
        pausedRef.current ||
        document.hidden ||
        !mounted.current ||
        lifecycle.current?.signal.aborted
      )
        return;
      if (!proposal.action) {
        report(proposal.explanation);
        return;
      }
      if (!actionSupported(proposal.action, capabilities, nativeActions))
        throw new Error("The interpreted action is unsupported by this worker.");
      await submit({
        type: proposal.action.type,
        action: proposal.action.action ?? undefined,
        prompt: proposal.action.prompt ?? undefined,
      });
      report(`Scene-aware command sent. ${proposal.explanation}`);
    } catch (failure) {
      if (!controller.signal.aborted) {
        setPrompt(text);
        report(errorText(failure));
      }
    } finally {
      if (voiceRequest.current === controller) voiceRequest.current = null;
    }
  }

  function startVoice() {
    if (listening) {
      voiceRequest.current?.abort();
      speech.current?.stop();
      return;
    }
    if (!voiceConsent) {
      setVoiceDialog(true);
      return;
    }
    const Speech =
      (window as SpeechWindow).SpeechRecognition ??
      (window as SpeechWindow).webkitSpeechRecognition;
    if (!Speech)
      return report("Speech recognition is unavailable in this browser. Use the command bar.");
    releaseInputs();
    const input = new VoiceInput();
    voiceRequest.current?.abort();
    speech.current?.stop();
    speech.current = input;
    setListening(true);
    setTranscript("Listening…");
    try {
      input.start(new Speech(), {
        continuous: voiceContinuous && voiceAutoSend,
        language: navigator.language,
        onTranscript: (text) => {
          if (mounted.current) setTranscript(text);
        },
        onCommand: (text) => {
          if (
            !mounted.current ||
            document.hidden ||
            pausedRef.current ||
            lifecycle.current?.signal.aborted
          )
            return;
          if (voiceAutoSend) {
            void sendVoice(text);
          } else {
            setPrompt(text);
            report("Voice transcribed. Review the command and press Enter to send.");
            inputRef.current?.focus();
          }
        },
        onError: (message) => report(`Voice input: ${message}. You can still type your command.`),
        onEnd: () => {
          if (mounted.current) setListening(false);
        },
      });
    } catch (error) {
      report(errorText(error));
    }
  }
  function persistBindings(next: ActionBinding[]) {
    setBindings(next);
    try {
      localStorage.setItem(
        `worlds-bindings:${preview ? "preview" : project.modelId}`,
        JSON.stringify(next),
      );
    } catch {
      report("Bindings work for this session, but browser storage is unavailable.");
    }
  }
  async function generateBindings() {
    if (!model) return;
    setBusy(true);
    try {
      if (preview) {
        const previewModel = {
          ...model,
          nativeActions: ["forward", "backward", "left", "right"],
          capabilities: {
            ...model.capabilities,
            control: { ...model.capabilities.control, promptDuringRollout: true },
          },
        };
        persistBindings(suggestBindings(previewModel));
        report(
          "Local suggested bindings applied. This is a deterministic template, not an AI response.",
        );
      } else if (capabilities) {
        const generated = await createWorldApi(serverUrl).generateControls({
          prompt: activePrompt,
          modelId: project.modelId,
          capabilities,
          nativeActions,
        });
        const liveModel = { ...model, capabilities, nativeActions };
        persistBindings(
          generated.bindings.filter((binding) => bindingSupport(binding, liveModel).supported),
        );
        report(
          generated.source === "llm"
            ? "AI-generated controls applied. Review or edit the bindings before using them."
            : "Local suggested controls applied. No AI generation service was used.",
        );
      }
    } catch (error) {
      report(errorText(error));
    } finally {
      setBusy(false);
    }
  }
  function addBinding() {
    if (
      !bindingText.trim() ||
      !/^(Key[A-Z]|Digit[0-9]|Space|Arrow(Up|Down|Left|Right))$/.test(bindingKey)
    )
      return report("Choose a letter, digit, arrow key, or Space.");
    if (bindingType === "native" && !preview && !nativeActions.includes(bindingText))
      return report("Choose a native action advertised by this worker.");
    if (bindingType === "prompt" && !canPrompt) return report("Live prompting is unavailable.");
    const next = bindings.filter((item) => item.key !== bindingKey);
    next.push({
      id: newId(),
      key: bindingKey,
      label: bindingText,
      type: bindingType,
      ...(bindingType === "native" ? { action: bindingText } : { prompt: bindingText }),
      gamepadButton: gamepadButton === "" ? undefined : Number(gamepadButton),
      experimental: bindingType !== "native",
    });
    persistBindings(next);
    report("Binding saved. Focus the world to use it.");
  }
  async function replayFile(file?: File) {
    if (!file) return;
    try {
      if (file.size > 10_000_000) throw new Error("Trajectory files must be smaller than 10 MB.");
      const trajectory = parseTrajectory(await file.text());
      if (
        trajectory.events.some(
          (event) =>
            event.type === "native" && !preview && !nativeActions.includes(event.action ?? ""),
        )
      )
        throw new Error("This trajectory contains actions unavailable on the connected worker.");
      const controller = new AbortController();
      trajectoryController.current?.abort();
      trajectoryController.current = controller;
      setReplaying(true);
      releaseInputs();
      await replayTrajectory(
        trajectory.events,
        (event) =>
          submit({
            type: event.type,
            action: event.action,
            prompt: event.prompt,
            values: event.values,
          }),
        { signal: controller.signal },
      );
      report(
        "Control trajectory complete. Identical inputs do not guarantee identical model output.",
      );
    } catch (error) {
      if (!(error instanceof DOMException && error.name === "AbortError")) report(errorText(error));
    } finally {
      releaseInputs();
      setReplaying(false);
    }
  }
  async function exit() {
    voiceRequest.current?.abort();
    speech.current?.stop();
    setExiting(true);
    releaseInputs();
    if (recorder.current?.state === "recording") recorder.current.stop();
    try {
      if (!preview && hasFrame.current)
        await worldStore
          .put("benchmarks", {
            id: newId(),
            name: `${project.name} session`,
            projectId: project.id,
            modelId: project.modelId,
            providerId: project.providerId,
            createdAt: Date.now(),
            durationMs: performance.now() - started.current,
            frameCount: frameCount.current,
            measuredFPS:
              frameCount.current / Math.max(1, (performance.now() - started.current) / 1000),
            estimatedCostUSD: hourlyCost === null ? undefined : (hourlyCost * elapsed) / 3600,
            notes:
              "Received unique frame updates averaged across full session including startup. Not model generation FPS.",
            events: [...events.current],
          })
          .catch((error: unknown) =>
            report(`Session metrics could not be saved: ${errorText(error)}`),
          );
      await teardown();
      onExit();
    } catch (error) {
      setError(errorText(error));
      setExiting(false);
    }
  }

  return (
    <div className="world-player" ref={screenRef} tabIndex={-1}>
      <canvas
        ref={canvasRef}
        className="world-player-canvas"
        aria-label={
          preview
            ? "Procedural landscape interface preview, not AI generated"
            : "Live generated world output"
        }
        onClick={() => {
          lastActivity.current = performance.now();
          screenRef.current?.focus();
          void videoRef.current?.play().catch(() => {
            /* Session may already be closed. */
          });
          if (preview || capabilities?.control.mouseLook)
            void canvasRef.current?.requestPointerLock?.();
        }}
        onMouseMove={(event) => {
          if (document.pointerLockElement !== canvasRef.current || pausedRef.current) return;
          if (preview) {
            view.current.yaw += event.movementX * 0.002;
            view.current.pitch = Math.max(
              -1,
              Math.min(1, view.current.pitch + event.movementY * 0.003),
            );
          } else if (capabilities?.control.mouseLook) {
            const delta = mouseDelta.current;
            delta.x += event.movementX;
            delta.y += event.movementY;
            if (performance.now() - delta.sentAt > 75) {
              void submit({
                type: "native",
                action: "look",
                values: { x: delta.x, y: delta.y },
              }).catch((error: unknown) => report(errorText(error)));
              delta.x = 0;
              delta.y = 0;
              delta.sentAt = performance.now();
            }
          }
        }}
      />
      <video ref={videoRef} className="world-player-video" playsInline autoPlay muted={muted}>
        <track kind="captions" />
      </video>
      <div className="world-player-shade" />
      <header className="world-player-top">
        <button
          className="wp-icon"
          aria-label="Exit world"
          title="End session and exit"
          onClick={() => void exit()}
          disabled={exiting}
        >
          <ArrowLeft size={19} />
        </button>
        <div className="wp-world-name">
          <strong>{project.name}</strong>
          <span>{preview ? "INTERFACE PREVIEW" : (model?.name ?? project.modelId)}</span>
        </div>
        <div className="wp-session-stats">
          <span className={`wp-live-dot ${paused ? "is-paused" : ""}`} />
          {paused
            ? "Paused"
            : preview
              ? "Preview"
              : status.startsWith("Live")
                ? "Live"
                : "Connecting"}
          <span className="wp-stat-divider" />
          <span>
            {fps} {preview ? "render" : "received"} FPS
          </span>
          <span className="wp-stat-divider" />
          <span>{formatTime(elapsed)}</span>
        </div>
        <button
          className="wp-icon"
          aria-label="Session details"
          onClick={() => {
            releaseInputs();
            setPanel(panel === "session" ? null : "session");
          }}
        >
          <Settings2 size={19} />
        </button>
        {!preview && (
          <button
            className="wp-icon"
            aria-label="AI director"
            title="Scene-aware commands and game director"
            onClick={() => {
              releaseInputs();
              setPanel(panel === "director" ? null : "director");
            }}
          >
            <Sparkles size={19} />
          </button>
        )}
        <button
          className="wp-icon wp-fullscreen"
          aria-label="Toggle fullscreen"
          onClick={() => {
            if (document.fullscreenElement) void document.exitFullscreen();
            else
              void screenRef.current
                ?.requestFullscreen()
                .catch((error: unknown) => report(errorText(error)));
          }}
        >
          <Expand size={19} />
        </button>
      </header>
      {preview && (
        <div className="wp-preview-label">
          <span>LOCAL PREVIEW</span> A procedural canvas for trying controls, saves & recording. No
          AI inference.
        </div>
      )}
      {!preview && receivedFrames === 0 && !error && (
        <div className="wp-connection">
          <div className="wp-orbit" />
          <h2>{status}</h2>
          <p>Waiting for your worker’s real output. The first frame will appear here.</p>
        </div>
      )}
      {error && (
        <div className="wp-error" role="alert">
          <span>SESSION NEEDS ATTENTION</span>
          <h2>
            {status === "Session could not start"
              ? "Unable to enter this world"
              : "Connection issue"}
          </h2>
          <p>{error}</p>
          <button className="wp-primary" onClick={() => void exit()} disabled={exiting}>
            {exiting ? "Ending session…" : "Return to worlds"}
          </button>
        </div>
      )}
      {paused && !error && (
        <div className="wp-paused">
          <Pause size={22} /> Generation paused <span>Allocated GPU billing may continue.</span>
        </div>
      )}
      <div className="wp-crosshair" aria-hidden="true" />
      {notice && (
        <div className="wp-toast" role="status">
          <Check size={16} />
          {notice}
        </div>
      )}
      <div className="world-player-bottom">
        <div className="wp-toolbar">
          <button
            className="wp-pill"
            onClick={() => {
              releaseInputs();
              setPanel(panel === "controls" ? null : "controls");
            }}
          >
            <Keyboard size={17} />
            <span>Controls</span>
          </button>
          <span className="wp-control-hint">
            {(preview || capabilities?.control.wasd) && (
              <>
                W A S D <span>{promptAdapted ? "guide movement" : "move"}</span> ·{" "}
              </>
            )}
            {(preview || capabilities?.control.mouseLook) && (
              <>
                click <span>{promptAdapted ? "guide view" : "look"}</span> ·{" "}
              </>
            )}
            Esc <span>release</span>
          </span>
          <div className="wp-toolbar-spacer" />
          <button
            className="wp-icon"
            aria-label={paused ? "Resume generation" : "Pause generation"}
            title={paused ? "Resume" : "Pause generation"}
            onClick={() => void togglePause()}
            disabled={!preview && !connected}
          >
            {paused ? <Play size={18} /> : <Pause size={18} />}
          </button>
          {capabilities?.output.audio && (
            <button
              className="wp-icon"
              aria-label={muted ? "Enable generated audio" : "Mute generated audio"}
              onClick={() => {
                setMuted(!muted);
                if (videoRef.current) {
                  videoRef.current.muted = !muted;
                  void videoRef.current
                    .play()
                    .catch(() => report("Click the world to enable audio playback."));
                }
              }}
            >
              {muted ? <VolumeX size={18} /> : <Volume2 size={18} />}
            </button>
          )}
          <button
            className="wp-icon"
            aria-label="Take screenshot"
            title="Screenshot"
            onClick={() => void screenshot()}
          >
            <Camera size={18} />
          </button>
          <button
            className={`wp-icon ${recording ? "wp-recording" : ""}`}
            aria-label={recording ? "Stop recording" : "Start recording"}
            title={recording ? "Stop & save recording" : "Record replay"}
            onClick={toggleRecording}
            disabled={recordingBusy}
          >
            {recording ? <Square size={16} fill="currentColor" /> : <Circle size={18} />}
          </button>
          <button className="wp-pill" onClick={() => void saveScene()} disabled={busy}>
            <Save size={16} />
            {busy ? "Saving…" : "Save scene"}
          </button>
        </div>
        {continuousChunks && nativeActions.includes("exploration") && (
          <ExplorationControls
            initialQuality={project.settings.performance}
            promptTruncated={revisions.promptTruncated}
            disabled={!connected || paused || exiting}
            onSubmit={submit}
          />
        )}
        {offlineClip && paused && (
          <label className="wp-next-clip">
            Next clip camera
            <select
              aria-label="Next clip camera"
              value={clipAction}
              onChange={(event) => setClipAction(event.target.value)}
            >
              {nativeActions.map((action) => (
                <option key={action} value={action}>
                  {action.replaceAll("_", " ")}
                </option>
              ))}
            </select>
            <span>Resume generates one clip with this direction and the current prompt.</span>
          </label>
        )}
        <form
          className="wp-command"
          onSubmit={(event) => {
            event.preventDefault();
            void sendPrompt(prompt);
          }}
        >
          <Command size={19} />
          <input
            ref={inputRef}
            value={prompt}
            onFocus={releaseInputs}
            onChange={(event) => setPrompt(event.target.value)}
            disabled={!canPrompt || (paused && !offlineClip)}
            maxLength={continuousChunks ? 3000 : 10000}
            aria-label="World command"
            placeholder={
              paused
                ? offlineClip
                  ? "Describe the next clip, then resume"
                  : "Resume generation to send a command"
                : canPrompt
                  ? preview
                    ? "Try “make it rain” or “turn the sky to sunset”…"
                    : "Describe what happens next…"
                  : "This worker does not advertise live prompt control"
            }
          />
          <button
            type="button"
            className={`wp-icon ${listening ? "wp-recording" : ""}`}
            disabled={!canPrompt || paused}
            aria-label={listening ? "Stop voice input" : "Use voice input"}
            onClick={startVoice}
          >
            <Mic size={18} />
          </button>
          <button
            className="wp-command-send"
            type="submit"
            disabled={!prompt.trim() || !canPrompt || (paused && !offlineClip)}
          >
            ↵
          </button>
        </form>
        <div className="wp-command-footer">
          <span>
            {listening
              ? transcript
              : preview
                ? "Your library stays in this browser"
                : continuousChunks
                  ? explorationProgress(revisions.queued, revisions.applied, revisions.generating)
                  : capabilities?.control.promptSwitching &&
                      !capabilities.control.promptDuringRollout
                    ? "Commands apply to the next generated clip"
                    : status}
          </span>
          <span>
            {hourlyCost === null
              ? "GPU cost unavailable"
              : preview
                ? "No GPU cost"
                : `~$${((hourlyCost * elapsed) / 3600).toFixed(3)} estimated`}{" "}
            {recording && " · ● REC"}
          </span>
        </div>
      </div>
      {panel && panel !== "director" && (
        <aside
          className="wp-panel"
          aria-label={panel === "controls" ? "Control bindings" : "Session details"}
        >
          <div className="wp-panel-heading">
            <div>
              <span>YOUR SESSION</span>
              <h2>{panel === "controls" ? "Make it your own" : "Behind the world"}</h2>
            </div>
            <button className="wp-icon" aria-label="Close panel" onClick={() => setPanel(null)}>
              <X size={19} />
            </button>
          </div>
          {panel === "controls" ? (
            <>
              <p className="wp-muted">
                {preview
                  ? "Preview bindings control this procedural canvas. Native model controls depend on the connected adapter."
                  : "Only the connected worker’s advertised capabilities are enabled. Prompt actions are experimental, not guaranteed game mechanics."}
              </p>
              <div className="wp-binding-list">
                {bindings.map((binding) => (
                  <div className="wp-binding" key={binding.id}>
                    <input
                      aria-label={`Key for ${binding.label}`}
                      value={binding.key.replace("Key", "").replace("Digit", "")}
                      onKeyDown={(event) => {
                        event.preventDefault();
                        if (
                          event.code !== "Escape" &&
                          !bindings.some(
                            (item) => item.id !== binding.id && item.key === event.code,
                          )
                        )
                          persistBindings(
                            bindings.map((item) =>
                              item.id === binding.id ? { ...item, key: event.code } : item,
                            ),
                          );
                      }}
                      readOnly
                    />
                    <span>
                      {binding.label}
                      <small>
                        {preview && binding.type === "native"
                          ? "PREVIEW"
                          : binding.type === "native" && promptAdapted
                            ? "PROMPT-ADAPTED"
                            : binding.type.toUpperCase()}
                        {binding.gamepadButton !== undefined
                          ? ` · PAD ${binding.gamepadButton}`
                          : ""}
                      </small>
                    </span>
                    <button
                      className="wp-icon"
                      aria-label={`Remove ${binding.label} binding`}
                      onClick={() =>
                        persistBindings(bindings.filter((item) => item.id !== binding.id))
                      }
                    >
                      <X size={14} />
                    </button>
                  </div>
                ))}
              </div>
              {!bindings.length && (
                <p className="wp-muted">
                  No movement controls advertised. Add a supported prompt binding below.
                </p>
              )}
              <button
                className="wp-pill"
                onClick={() => void generateBindings()}
                disabled={busy || (!preview && !capabilities)}
              >
                {busy ? "Preparing controls…" : preview ? "Suggest controls" : "Generate controls"}
              </button>
              <h3>Voice commands</h3>
              <button
                className="wp-pill"
                onClick={() => {
                  voiceRequest.current?.abort();
                  speech.current?.stop();
                  setVoiceDialog(true);
                }}
              >
                Voice settings
              </button>
              <h3>Add a binding</h3>
              <div className="wp-bind-editor">
                <label>
                  Key
                  <input
                    value={bindingKey.replace("Key", "").replace("Digit", "")}
                    aria-label="New binding key"
                    onKeyDown={(event) => {
                      event.preventDefault();
                      setBindingKey(event.code);
                    }}
                    readOnly
                  />
                </label>
                <label>
                  Gamepad button
                  <select
                    value={gamepadButton}
                    onChange={(event) => setGamepadButton(event.target.value)}
                  >
                    <option value="">None</option>
                    {Array.from({ length: 16 }, (_, i) => (
                      <option key={i} value={i}>
                        {i}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <label className="wp-field">
                Mechanism
                <select
                  value={bindingType}
                  onChange={(event) => setBindingType(event.target.value as typeof bindingType)}
                >
                  <option value="prompt" disabled={!canPrompt}>
                    Prompt · experimental
                  </option>
                  <option value="native" disabled={!preview && !nativeActions.length}>
                    {promptAdapted ? "Prompt-adapted movement" : "Native action"}
                  </option>
                  <option
                    value="semantic"
                    disabled={!preview && !capabilities?.control.semanticActions}
                  >
                    Semantic action
                  </option>
                </select>
              </label>
              {bindingType === "native" ? (
                <label className="wp-field">
                  Action
                  <select
                    value={bindingText}
                    onChange={(event) => setBindingText(event.target.value)}
                  >
                    <option value="">Choose an advertised action</option>
                    {(preview
                      ? ["forward", "backward", "left", "right"]
                      : nativeActions.filter(
                          (action) => !["exploration", "look", "analog"].includes(action),
                        )
                    ).map((action) => (
                      <option key={action}>{action}</option>
                    ))}
                  </select>
                </label>
              ) : (
                <label className="wp-field">
                  Instruction
                  <textarea
                    value={bindingText}
                    onChange={(event) => setBindingText(event.target.value)}
                    maxLength={2000}
                    rows={3}
                  />
                </label>
              )}
              <button className="wp-primary" onClick={addBinding}>
                Save binding
              </button>
              <p className="wp-muted">
                Click a key field and press a key to remap. Gamepad buttons work when the world has
                focus. Press / to open the command bar.
              </p>
            </>
          ) : (
            <>
              <div className="wp-details">
                <span>Model</span>
                <strong>{model?.name ?? project.modelId}</strong>
                <span>Compute</span>
                <strong>{preview ? "Browser · no inference" : project.providerId}</strong>
                <span>Stream</span>
                <strong>{preview ? "Procedural canvas" : status}</strong>
                <span>Frames received</span>
                <strong>{receivedFrames.toLocaleString()}</strong>
                {!preview && (
                  <>
                    <span>Generated FPS · last block</span>
                    <strong>
                      {generationMetrics.generatedFPS?.toFixed(2) ?? "Waiting for measured output"}
                    </strong>
                    <span>Model loads / load time</span>
                    <strong>
                      {generationMetrics.modelLoadCount ?? "—"} /{" "}
                      {generationMetrics.modelLoadSeconds === undefined
                        ? "—"
                        : `${generationMetrics.modelLoadSeconds.toFixed(1)}s`}
                    </strong>
                    <span>Video codec / encoder</span>
                    <strong>
                      {streamMetrics.codec ?? "Frame transport"} / {streamMetrics.encoder ?? "—"}
                    </strong>
                    <span>Control round trip</span>
                    <strong>
                      {streamMetrics.controlLatencyMs === undefined
                        ? "Not measured"
                        : `${streamMetrics.controlLatencyMs.toFixed(0)} ms`}
                    </strong>
                    <span>Network round trip</span>
                    <strong>
                      {streamMetrics.networkRttMs === undefined
                        ? "Not measured"
                        : `${streamMetrics.networkRttMs.toFixed(0)} ms`}
                    </strong>
                    <span>Dropped transport frames</span>
                    <strong>{streamMetrics.workerDroppedFrames ?? "Not reported"}</strong>
                    {streamMetrics.fallbackReason && (
                      <>
                        <span>Compatibility fallback</span>
                        <strong>{streamMetrics.fallbackReason}</strong>
                      </>
                    )}
                  </>
                )}
                <span>Elapsed</span>
                <strong>{formatTime(elapsed)}</strong>
                <span>Cost / hour</span>
                <strong>
                  {hourlyCost === null
                    ? "Not reported by provider"
                    : `$${hourlyCost.toFixed(2)}${preview ? "" : " estimated"}`}
                </strong>
                <span>Seed</span>
                <strong>{project.settings.seed ?? "Not specified"}</strong>
              </div>
              <p className="wp-muted">
                {preview
                  ? "This preview tests the product interface. It does not measure model quality or inference performance."
                  : `${workerManaged ? (settings.retainWorker ? "Worker retention is enabled. Your worker remains allocated after exit and may continue billing." : "The session owns this worker and requests teardown on exit.") : "Externally managed workers may continue billing after disconnect. Stop them in your provider console."} Pausing generation does not stop GPU billing. FPS measures received video frames, not inferred frames.`}
              </p>
              <h3>Current instruction</h3>
              <p className="wp-current-prompt">{activePrompt}</p>
              <h3>Prompt trajectory</h3>
              <div className="wp-event-list">
                {recentEvents.length ? (
                  recentEvents.map((event) => (
                    <div key={event.id}>
                      <time>{formatTime(event.timestampMs / 1000)}</time>
                      <span>{event.prompt ?? event.type}</span>
                    </div>
                  ))
                ) : (
                  <p className="wp-muted">Your commands will appear here.</p>
                )}
              </div>
              <button
                className="wp-pill"
                onClick={() =>
                  saveDownload(exportTrajectory(project, events.current), "world-trajectory.json")
                }
              >
                <Download size={15} />
                Export control trajectory
              </button>
              <label className="wp-field">
                Replay control trajectory
                <input
                  type="file"
                  accept="application/json,.json"
                  disabled={replaying}
                  onChange={(event) => {
                    void replayFile(event.target.files?.[0]);
                    event.target.value = "";
                  }}
                />
              </label>
              {replaying && (
                <button className="wp-pill" onClick={() => trajectoryController.current?.abort()}>
                  Stop trajectory replay
                </button>
              )}
              <p className="wp-muted">
                Replays the original event timing against this session. Output is not guaranteed
                deterministic; unsupported actions stop the replay.
              </p>
            </>
          )}
        </aside>
      )}
      {!preview && capabilities && (
        <aside className="wp-panel" aria-label="AI game director" hidden={panel !== "director"}>
          <div className="wp-panel-heading">
            <h2>Guide the world</h2>
            <button className="wp-icon" aria-label="Close director" onClick={() => setPanel(null)}>
              <X size={19} />
            </button>
          </div>
          <IntelligencePanel
            serverUrl={serverUrl}
            modelId={project.modelId}
            capabilities={capabilities}
            nativeActions={nativeActions}
            worldPrompt={activePrompt}
            getFrame={currentFrame}
            onCommand={submit}
            events={recentEvents}
            elapsedSeconds={elapsed}
            sessionKey={project.id}
            initialObjective={project.game?.objective}
            scheduledEvents={project.game?.events}
            disabled={paused || exiting || !connected || !!error || replaying}
          />
        </aside>
      )}
      {voiceDialog && (
        <div className="wp-modal-scrim">
          <div
            className="wp-voice-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="voice-title"
          >
            <Mic size={28} />
            <h2 id="voice-title">Control your world by voice</h2>
            <p>
              This uses your browser’s speech recognition. Your browser may send microphone audio to
              its speech service; this is not guaranteed to stay on your device.
            </p>
            <p>
              We do not record your microphone. Final spoken phrases can go directly to the world.
              Changes take effect when the connected model processes its next input.
            </p>
            <label className="wp-voice-option">
              <input
                type="checkbox"
                checked={voiceAutoSend}
                onChange={(event) => setVoiceAutoSend(event.target.checked)}
              />
              Send spoken commands immediately
            </label>
            <label className="wp-voice-option">
              <input
                type="checkbox"
                checked={voiceContinuous}
                onChange={(event) => setVoiceContinuous(event.target.checked)}
                disabled={!voiceAutoSend}
              />
              Keep listening between commands
            </label>
            {!preview && (
              <label className="wp-voice-option">
                <input
                  type="checkbox"
                  checked={voiceSceneAware}
                  onChange={(event) => setVoiceSceneAware(event.target.checked)}
                  disabled={!voiceAutoSend}
                />
                Interpret commands with the current scene
              </label>
            )}
            {voiceSceneAware && !preview && (
              <p>
                Scene-aware mode sends a current frame and recent commands to your configured vision
                service. It adds interpretation latency.
              </p>
            )}
            <p>Listening stops when you pause, leave this tab, or press the microphone again.</p>
            <div>
              <button className="wp-pill" onClick={() => setVoiceDialog(false)}>
                Cancel
              </button>
              <button
                className="wp-primary"
                onClick={() => {
                  setVoiceConsent(true);
                  setVoiceDialog(false);
                  report(
                    "Voice enabled for this session. Click the microphone to start listening.",
                  );
                }}
              >
                Enable voice
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
