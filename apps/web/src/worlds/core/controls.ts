import {
  newId,
  type ActionBinding,
  type ControlEvent,
  type ModelInfo,
  type WorldProject,
} from "./types";
/** Deterministic suggestions, not AI-generated controls. UI must label accordingly. */
export function suggestBindings(model: ModelInfo): ActionBinding[] {
  const bindings: ActionBinding[] = [];
  const keys = ["KeyW", "KeyS", "KeyA", "KeyD", "Space", "KeyE", "KeyF", "KeyQ"];
  model.nativeActions
    .filter((action) => !["exploration", "look", "analog"].includes(action))
    .slice(0, keys.length)
    .forEach((action, index) => {
      bindings.push({
        id: newId(),
        key: keys[index] ?? "Unidentified",
        label: action.replaceAll("_", " "),
        type: "native",
        action,
      });
    });
  if (
    model.capabilities.control.semanticActions ||
    model.capabilities.control.promptDuringRollout ||
    model.capabilities.control.promptSwitching
  ) {
    const type = model.capabilities.control.semanticActions ? "semantic" : "prompt";
    if (!bindings.some((binding) => binding.key === "KeyE"))
      bindings.push({
        id: newId(),
        key: "KeyE",
        label: "Interact",
        type,
        prompt: "Interact with the object directly in front of the viewpoint.",
        experimental: true,
      });
    if (!bindings.some((binding) => binding.key === "KeyF"))
      bindings.push({
        id: newId(),
        key: "KeyF",
        label: "Change weather",
        type,
        prompt: "Rain begins to fall while preserving the current scene and viewpoint.",
        experimental: true,
      });
  }
  return bindings;
}
export function bindingEvent(
  binding: ActionBinding,
  timestampMs: number,
  pressed = true,
): ControlEvent {
  return {
    id: newId(),
    timestampMs: Math.max(0, timestampMs),
    type: binding.type,
    action: binding.action,
    prompt: binding.prompt,
    values: binding.type === "native" ? { pressed } : undefined,
  };
}
export function bindingSupport(
  binding: ActionBinding,
  model: ModelInfo,
): { supported: boolean; reason?: string } {
  if (binding.type === "native")
    return model.nativeActions.includes(binding.action ?? "")
      ? { supported: true }
      : { supported: false, reason: "This native action is not in the model adapter vocabulary." };
  if (binding.type === "semantic")
    return model.capabilities.control.semanticActions
      ? { supported: true }
      : { supported: false, reason: "This adapter does not expose semantic actions." };
  return model.capabilities.control.promptDuringRollout ||
    model.capabilities.control.promptSwitching
    ? { supported: true }
    : {
        supported: false,
        reason: "This adapter does not support prompt changes during generation.",
      };
}
export function promptSuggestion(
  prompt: string,
  model: ModelInfo,
  viewpoint = "first-person",
): { original: string; enhanced: string; source: "local-template" } {
  const original = prompt.trim();
  const camera = model.capabilities.control.camera6DoF
    ? "Maintain coherent camera motion and clear nearby and distant landmarks."
    : "Keep a consistent viewpoint, lighting, and visual identity across successive frames.";
  return {
    original,
    enhanced: `${original}${/[.!?]$/.test(original) ? "" : "."} ${viewpoint} view, natural visual scale, clear environmental details and atmosphere. ${camera}`,
    source: "local-template",
  };
}
export interface ControlTrajectory {
  format: "worlds-control-trajectory";
  version: 1;
  prompt: string;
  seed?: number;
  events: ControlEvent[];
}
export function exportTrajectory(project: WorldProject, events: ControlEvent[]): Blob {
  const data: ControlTrajectory = {
    format: "worlds-control-trajectory",
    version: 1,
    prompt: project.prompt,
    seed: project.settings.seed,
    events: [...events].sort((a, b) => a.timestampMs - b.timestampMs),
  };
  return new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
}
/** Preserves relative event timing; unsupported actions are reported, never silently remapped. */
export async function replayTrajectory(
  events: ControlEvent[],
  send: (event: ControlEvent) => Promise<unknown>,
  options: { signal?: AbortSignal; speed?: number; onProgress?: (index: number) => void } = {},
): Promise<void> {
  const speed = options.speed ?? 1;
  if (!Number.isFinite(speed) || speed <= 0 || speed > 10)
    throw new Error("Replay speed must be between 0 and 10.");
  const ordered = [...events].sort((a, b) => a.timestampMs - b.timestampMs);
  const started = performance.now();
  for (let index = 0; index < ordered.length; index++) {
    const event = ordered[index];
    if (!event) continue;
    if (!Number.isFinite(event.timestampMs) || event.timestampMs < 0)
      throw new Error("Trajectory contains an invalid timestamp.");
    options.signal?.throwIfAborted();
    const delay = event.timestampMs / speed - (performance.now() - started);
    if (delay > 0)
      await new Promise<void>((resolve, reject) => {
        const finish = () => {
          options.signal?.removeEventListener("abort", abort);
          resolve();
        };
        const timer = setTimeout(finish, delay);
        const abort = () => {
          clearTimeout(timer);
          options.signal?.removeEventListener("abort", abort);
          reject(new DOMException("Trajectory replay cancelled.", "AbortError"));
        };
        options.signal?.addEventListener("abort", abort, { once: true });
      });
    options.signal?.throwIfAborted();
    await send(event);
    options.onProgress?.(index + 1);
  }
}
export function parseTrajectory(json: string): ControlTrajectory {
  if (json.length > 10_000_000) throw new Error("Control trajectory is too large.");
  const data: unknown = JSON.parse(json);
  if (!data || typeof data !== "object") throw new Error("Invalid control trajectory.");
  const record = data as Partial<ControlTrajectory>;
  if (
    record.format !== "worlds-control-trajectory" ||
    record.version !== 1 ||
    typeof record.prompt !== "string" ||
    !Array.isArray(record.events) ||
    record.events.length > 100_000
  )
    throw new Error("Unsupported control trajectory.");
  for (const event of record.events) {
    if (
      !event ||
      typeof event.id !== "string" ||
      !Number.isFinite(event.timestampMs) ||
      event.timestampMs < 0 ||
      !["native", "prompt", "semantic", "pause", "resume"].includes(event.type)
    )
      throw new Error("Invalid event in control trajectory.");
    if (
      (event.prompt !== undefined && typeof event.prompt !== "string") ||
      (event.action !== undefined && typeof event.action !== "string")
    )
      throw new Error("Invalid action payload.");
    if (
      event.values !== undefined &&
      (typeof event.values !== "object" ||
        event.values === null ||
        Array.isArray(event.values) ||
        Object.values(event.values).some(
          (value) =>
            !["string", "number", "boolean"].includes(typeof value) ||
            (typeof value === "number" && !Number.isFinite(value)),
        ))
    )
      throw new Error("Invalid action vector.");
  }
  if (record.seed !== undefined && (!Number.isSafeInteger(record.seed) || record.seed < 0))
    throw new Error("Invalid trajectory seed.");
  return record as ControlTrajectory;
}
