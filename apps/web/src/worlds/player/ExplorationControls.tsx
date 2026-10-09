import { useState } from "react";
import type { ControlEvent, PerformanceMode } from "../core/types";

export interface ExplorationSettings {
  mode: "walk" | "direct" | "cruise";
  cadence: "responsive" | "balanced" | "smooth";
  speed: number;
}

export function ExplorationControls({
  disabled,
  initialQuality,
  promptTruncated,
  onSubmit,
}: {
  disabled: boolean;
  initialQuality: PerformanceMode;
  promptTruncated?: boolean;
  onSubmit: (event: Omit<ControlEvent, "id" | "timestampMs">) => Promise<void>;
}) {
  const [settings, setSettings] = useState<ExplorationSettings>({
    mode: "walk",
    cadence:
      initialQuality === "low-latency"
        ? "responsive"
        : initialQuality === "quality"
          ? "smooth"
          : "balanced",
    speed: 0.5,
  });
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  return (
    <form
      className="wp-exploration"
      aria-label="Exploration settings"
      onSubmit={(event) => {
        event.preventDefault();
        setBusy(true);
        setMessage("");
        void onSubmit({ type: "native", action: "exploration", values: { ...settings } })
          .then(() => setMessage("Settings received · apply to the next chunk"))
          .catch((error: unknown) =>
            setMessage(error instanceof Error ? error.message : "Settings could not be sent."),
          )
          .finally(() => setBusy(false));
      }}
    >
      <fieldset disabled={disabled || busy}>
        <label>
          Explore
          <select
            aria-label="Exploration mode"
            value={settings.mode}
            onChange={(event) => {
              setSettings({ ...settings, mode: event.target.value as ExplorationSettings["mode"] });
              setMessage("");
            }}
          >
            <option value="walk">Walk · keyboard / mouse</option>
            <option value="direct">Direct · stationary position</option>
            <option value="cruise">Cruise · automatic forward</option>
          </select>
        </label>
        <label>
          Cadence
          <select
            aria-label="Generation cadence"
            value={settings.cadence}
            onChange={(event) => {
              setSettings({
                ...settings,
                cadence: event.target.value as ExplorationSettings["cadence"],
              });
              setMessage("");
            }}
          >
            <option value="responsive">Responsive · shorter chunks</option>
            <option value="balanced">Balanced</option>
            <option value="smooth">Smooth · longer chunks</option>
          </select>
        </label>
        <label>
          Movement {Math.round(settings.speed * 100)}%
          <input
            aria-label="Exploration movement speed"
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={settings.speed}
            onChange={(event) => {
              setSettings({
                ...settings,
                speed: Math.max(0, Math.min(1, Number(event.target.value))),
              });
              setMessage("");
            }}
          />
        </label>
        <button className="wp-pill" type="submit">
          {busy ? "Sending…" : "Apply exploration settings"}
        </button>
      </fieldset>
      <p>
        {message ||
          "Prompt-adapted movement · changes apply between chunks. Cadence trades responsiveness for continuity; GPU speed is unverified."}
      </p>
      {promptTruncated && (
        <p>Scene context shortened to fit model; newest directions prioritized.</p>
      )}
    </form>
  );
}
