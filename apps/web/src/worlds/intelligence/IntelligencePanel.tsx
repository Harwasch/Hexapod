import { useCallback, useEffect, useRef, useState } from "react";
import { Eye, Loader2, Play, Sparkles, Square } from "lucide-react";
import type { ControlEvent, ModelCapabilities } from "../core/types";
import {
  actionSupported,
  getIntelligenceStatus,
  interpretCommand,
  observeWorld,
  type CommandProposal,
  type DirectorProposal,
  type IntelligenceStatus,
  type ProposedAction,
  type SceneContext,
} from "./client";
import "./intelligence.css";
export interface IntelligencePanelProps {
  serverUrl: string;
  modelId: string;
  capabilities: ModelCapabilities;
  nativeActions: string[];
  worldPrompt: string;
  getFrame: () => Promise<Blob | undefined>;
  onCommand: (action: {
    type: "native" | "prompt" | "semantic";
    action?: string;
    prompt?: string;
  }) => Promise<void>;
  events?: ControlEvent[];
  elapsedSeconds?: number;
  disabled?: boolean;
  sessionKey?: string;
  initialObjective?: string;
  scheduledEvents?: { atSeconds: number; prompt: string }[];
}
/** Mount for the session lifetime (hide with CSS if desired) so director cadence and review persist. */
export function IntelligencePanel(props: IntelligencePanelProps) {
  const { serverUrl, modelId, worldPrompt, disabled = false, initialObjective = "" } = props;
  const [status, setStatus] = useState<IntelligenceStatus>();
  const [consent, setConsent] = useState(false);
  const [command, setCommand] = useState("");
  const [objective, setObjective] = useState(initialObjective);
  const [mode, setMode] = useState<"relaxed" | "cinematic" | "challenging" | "chaotic">(
    "cinematic",
  );
  const [interval, setIntervalSeconds] = useState(60);
  const [running, setRunning] = useState(false);
  const [automatic, setAutomatic] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [proposal, setProposal] = useState<CommandProposal>();
  const [director, setDirector] = useState<DirectorProposal>();
  const latest = useRef(props);
  const options = useRef({ consent, objective, mode, automatic });
  const active = useRef<AbortController | null>(null);
  const occupied = useRef(false);
  const generation = useRef(0);
  const fired = useRef(new Set<number>());
  const lastEventAt = useRef(0);
  const proposalEpoch = useRef(-1);
  const directorEpoch = useRef(-1);
  useEffect(() => {
    latest.current = props;
    options.current = { consent, objective, mode, automatic };
  });
  useEffect(() => {
    const controller = new AbortController();
    void getIntelligenceStatus(serverUrl, controller.signal)
      .then(setStatus)
      .catch((failure: unknown) => {
        if (!controller.signal.aborted)
          setError(
            failure instanceof Error ? failure.message : "Intelligence service is unavailable.",
          );
      });
    return () => controller.abort();
  }, [serverUrl]);
  useEffect(() => {
    generation.current += 1;
    active.current?.abort();
    occupied.current = false;
    return () => {
      generation.current += 1;
      active.current?.abort();
    };
  }, [disabled, modelId, worldPrompt, serverUrl, props.sessionKey]);
  const apply = useCallback(async (action: ProposedAction, version: number) => {
    const p = latest.current;
    if (
      version !== generation.current ||
      p.disabled ||
      !options.current.consent ||
      !actionSupported(action, p.capabilities, p.nativeActions)
    )
      throw new Error(
        "This action is stale, paused, or unsupported. Observe the current scene again.",
      );
    lastEventAt.current = Date.now();
    await p.onCommand({
      type: action.type,
      ...(action.action ? { action: action.action } : {}),
      ...(action.prompt ? { prompt: action.prompt } : {}),
    });
    lastEventAt.current = Date.now();
    setMessage("Instruction sent to the world model. Its response may be delayed or ignored.");
  }, []);
  const execute = useCallback(
    async (kind: "command" | "director", text = "") => {
      if (occupied.current || latest.current.disabled || !options.current.consent) return;
      occupied.current = true;
      setBusy(true);
      setError("");
      setMessage("");
      const version = generation.current;
      const controller = new AbortController();
      active.current = controller;
      try {
        const p = latest.current;
        const frame = await p.getFrame();
        if (!frame)
          throw new Error("A current world frame is required. Wait for the model output.");
        const context: SceneContext = {
          serverUrl: p.serverUrl,
          modelId: p.modelId,
          capabilities: p.capabilities,
          nativeActions: p.nativeActions,
          prompt: p.worldPrompt,
          frame,
          objective: options.current.objective,
          elapsedSeconds: p.elapsedSeconds,
          recentEvents: p.events?.slice(-20).map((e) => e.prompt ?? e.action ?? e.type),
          signal: controller.signal,
        };
        if (kind === "command") {
          const result = await interpretCommand({ ...context, command: text });
          if (version !== generation.current) return;
          proposalEpoch.current = version;
          setProposal(result);
        } else {
          const result = await observeWorld({ ...context, mode: options.current.mode });
          if (version !== generation.current) return;
          directorEpoch.current = version;
          setDirector(result);
          if (
            options.current.automatic &&
            result.event &&
            Date.now() - lastEventAt.current >= 30_000
          )
            await apply(result.event, version);
        }
      } catch (failure) {
        if (!controller.signal.aborted && version === generation.current)
          setError(failure instanceof Error ? failure.message : "AI observation failed.");
      } finally {
        if (active.current === controller) {
          occupied.current = false;
          setBusy(false);
          active.current = null;
        }
      }
    },
    [apply],
  );
  useEffect(() => {
    if (!running || disabled || !consent || !status?.visionConfigured) return;
    const timer = window.setInterval(
      () => {
        if (document.visibilityState === "visible") void execute("director");
      },
      Math.max(30, interval) * 1000,
    );
    return () => window.clearInterval(timer);
  }, [running, disabled, consent, interval, status?.visionConfigured, execute]);
  useEffect(() => {
    if (
      !running ||
      !automatic ||
      !consent ||
      disabled ||
      document.visibilityState !== "visible" ||
      occupied.current
    )
      return;
    const now = props.elapsedSeconds ?? 0;
    const event = props.scheduledEvents?.find(
      (item, index) => item.atSeconds <= now && !fired.current.has(index),
    );
    if (!event || Date.now() - lastEventAt.current < 30_000) return;
    const index = props.scheduledEvents?.indexOf(event);
    if (index === undefined) return;
    fired.current.add(index);
    const version = generation.current;
    void apply({ type: "prompt", prompt: event.prompt }, version).catch((failure: unknown) =>
      setError(failure instanceof Error ? failure.message : "Scheduled event could not be sent."),
    );
  }, [props.elapsedSeconds, props.scheduledEvents, running, automatic, consent, disabled, apply]);
  const manualApply = (action: ProposedAction, epoch: number) => {
    setError("");
    void apply(action, epoch).catch((failure: unknown) =>
      setError(failure instanceof Error ? failure.message : "Instruction failed."),
    );
  };
  return (
    <section className="wi-panel" aria-label="AI director and commands">
      <h3>
        <Sparkles size={16} />
        Scene intelligence
      </h3>
      <p className="wi-muted">
        A configured vision model observes a current frame. Guidance and progress are
        interpretations, not authoritative game state.
      </p>
      {!status?.visionConfigured && (
        <p className="wi-notice">
          Configure WORLDS_VISION_BASE_URL and WORLDS_VISION_MODEL on the session manager to enable
          frame-aware assistance.
        </p>
      )}
      <label className="wi-check">
        <input
          type="checkbox"
          checked={consent}
          onChange={(e) => {
            setConsent(e.target.checked);
            if (!e.target.checked) {
              setRunning(false);
              generation.current += 1;
              active.current?.abort();
            }
          }}
        />
        Send current frames and required scene context to the configured vision service.
      </label>
      <label className="wi-field">
        Describe an action
        <input
          value={command}
          maxLength={1000}
          onChange={(e) => setCommand(e.target.value)}
          placeholder="Pick up the red object in front of me"
        />
      </label>
      <button
        type="button"
        disabled={disabled || busy || !consent || !status?.visionConfigured || !command.trim()}
        onClick={() => void execute("command", command)}
      >
        <Eye size={14} />
        Interpret from this frame
      </button>
      {proposal && (
        <div className="wi-result">
          <p>{proposal.observation}</p>
          <p className="wi-muted">{proposal.explanation}</p>
          {proposal.action && (
            <>
              <code>{proposal.action.action ?? proposal.action.prompt}</code>
              <button
                type="button"
                disabled={disabled || !consent || busy}
                onClick={() => {
                  if (proposal.action) manualApply(proposal.action, proposalEpoch.current);
                }}
              >
                Apply suggested action
              </button>
            </>
          )}
        </div>
      )}
      <label className="wi-field">
        Narrative objective
        <textarea
          value={objective}
          maxLength={1000}
          onChange={(e) => setObjective(e.target.value)}
          placeholder="Reach the transmitter and inspect its light"
        />
      </label>
      <div className="wi-row">
        <label className="wi-field">
          Pacing
          <select value={mode} onChange={(e) => setMode(e.target.value as typeof mode)}>
            {["relaxed", "cinematic", "challenging", "chaotic"].map((v) => (
              <option key={v}>{v}</option>
            ))}
          </select>
        </label>
        <label className="wi-field">
          Observe every
          <select value={interval} onChange={(e) => setIntervalSeconds(Number(e.target.value))}>
            <option value={30}>30 seconds</option>
            <option value={60}>60 seconds</option>
            <option value={120}>2 minutes</option>
          </select>
        </label>
      </div>
      <label className="wi-check">
        <input
          type="checkbox"
          checked={automatic}
          onChange={(e) => setAutomatic(e.target.checked)}
        />
        Automatically send supported director events and scheduled story prompts (at most one every
        30 seconds).
      </label>
      <div className="wi-row">
        <button
          type="button"
          disabled={disabled || busy || !consent || !status?.visionConfigured}
          onClick={() => void execute("director")}
        >
          <Eye size={14} />
          Observe now
        </button>
        <button
          type="button"
          disabled={disabled || !consent || !status?.visionConfigured}
          onClick={() => {
            setRunning(!running);
            if (running) active.current?.abort();
          }}
        >
          {running ? <Square size={14} /> : <Play size={14} />}{" "}
          {running ? "Stop director" : "Start director"}
        </button>
      </div>
      <p className="wi-muted">
        {running
          ? disabled
            ? "Director paused with the session."
            : "Director enabled; observations pause while this tab is hidden."
          : "Director is off."}{" "}
        {automatic ? "Automatic events are enabled." : "Each proposed event waits for your review."}
      </p>
      {busy && (
        <p role="status">
          <Loader2 className="w-spin" size={14} /> Observing the current frame…
        </p>
      )}
      {director && (
        <div className="wi-result">
          <p>{director.observation}</p>
          <strong>Objective: {director.objective.progress.replaceAll("-", " ")}</strong>
          <p className="wi-muted">
            AI confidence {Math.round(director.objective.confidence * 100)}% ·{" "}
            {director.objective.evidence}
          </p>
          <p>{director.reason}</p>
          {director.event && (
            <>
              <code>{director.event.prompt ?? director.event.action}</code>
              <button
                type="button"
                disabled={disabled || busy || !consent}
                onClick={() => {
                  if (director.event) manualApply(director.event, directorEpoch.current);
                }}
              >
                Apply proposed event
              </button>
            </>
          )}
        </div>
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
