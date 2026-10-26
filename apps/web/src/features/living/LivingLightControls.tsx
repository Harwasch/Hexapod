import { useAsCaptured } from "@/state/asCaptured";
import { useLivingLight, type LightArm } from "@/state/livingLight";

export function LivingLightControls() {
  const light = useLivingLight();
  const captured = useAsCaptured((s) => s.on);
  return (
    <fieldset disabled={captured}>
      <legend>Light · Simulated</legend>
      <label className="living-lab__row">
        Arm{" "}
        <select value={light.arm} onChange={(e) => light.set({ arm: e.target.value as LightArm })}>
          <option value="off">Off (today’s relight)</option>
          <option value="a">A · de-lit albedo + sun / sky</option>
          <option value="b">B · A + visibility / clouds</option>
        </select>
      </label>
      <label className="living-lab__row">
        Solar day, 07–17 h{" "}
        <input
          type="range"
          min="0"
          max="1"
          step="0.005"
          value={light.day}
          onChange={(e) => light.set({ day: Number(e.target.value) })}
        />
      </label>
      <label className="living-lab__row">
        Cloud shadow{" "}
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={light.cloud}
          onChange={(e) => light.set({ cloud: Number(e.target.value) })}
        />
      </label>
      <small>
        {light.status}. Static scan, Ground truth look; local sidecar required via lightManifest.
      </small>
    </fieldset>
  );
}
