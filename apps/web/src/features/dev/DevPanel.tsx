import { useEffect, useState } from "react";

import { Divider, GlassSwitch } from "@twin/ui";

import { useScene } from "@/cesium/SceneContext";
import type { DebugFlags } from "@/cesium/DebugManager";
import { representationLabel } from "@/lib/format";
import { recentSpans, onSpan } from "@/lib/timing";
import { useSettings } from "@/state/settings";
import { useSites } from "@/state/sites";
import { useViewer } from "@/state/viewer";

import { FloatingPanel } from "../shell/FloatingPanel";

function Flag({
  id,
  label,
  checked,
  onChange,
}: {
  id: keyof DebugFlags;
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <div className="setting" style={{ padding: "0.3rem 0" }}>
      <span className="setting__label" id={`dev-${id}`}>
        {label}
      </span>
      <GlassSwitch aria-labelledby={`dev-${id}`} checked={checked} onCheckedChange={onChange} />
    </div>
  );
}

/** Developer panel: FPS, camera, tilesets, requests, SSE, debug volumes, GPU. */
export function DevPanel() {
  const scene = useScene();
  const open = useSettings((s) => s.devToolsOpen);
  const set = useSettings((s) => s.set);
  const camera = useViewer((s) => s.camera);
  const perf = useViewer((s) => s.performance);
  const tilesets = useViewer((s) => s.activeTilesets);
  const activeSiteId = useSites((s) => s.activeSiteId);
  const representation = useSites((s) =>
    activeSiteId ? s.representation[activeSiteId] : undefined,
  );
  const assets = useSites((s) => s.assets);
  const [flags, setFlags] = useState<DebugFlags>(
    () =>
      scene?.debug.current ?? {
        boundingVolumes: false,
        wireframe: false,
        tileCoordinates: false,
        fpsOverlay: false,
        clipping: true,
      },
  );
  const [apiMs, setApiMs] = useState<number | null>(() => {
    const last = recentSpans()
      .filter((s) => s.name === "api")
      .at(-1);
    return last ? Math.round(last.durationMs) : null;
  });
  useEffect(
    () => onSpan((span) => span.name === "api" && setApiMs(Math.round(span.durationMs))),
    [],
  );
  const setFlag = (patch: Partial<DebugFlags>) => {
    if (!scene) return;
    setFlags(scene.debug.set(patch));
  };
  const memory = Object.values(assets).reduce((sum, a) => sum + a.memoryMb, 0);
  const pending = Object.values(assets).reduce((sum, a) => sum + a.progress.pending, 0);
  return (
    <FloatingPanel
      open={open}
      from="right"
      title="Developer"
      onClose={() => set({ devToolsOpen: false })}
      testId="dev-panel"
    >
      <dl className="dev-grid">
        <dt>FPS</dt>
        <dd data-testid="dev-fps">{perf.rendering ? perf.fps : "idle"}</dd>
        <dt>Frame</dt>
        <dd>{perf.rendering ? `${perf.frameTimeMs} ms` : "—"}</dd>
        <dt>Altitude</dt>
        <dd>{camera.altitude.toFixed(1)} m</dd>
        <dt>Lat / Lon</dt>
        <dd>
          {camera.latitude.toFixed(6)}, {camera.longitude.toFixed(6)}
        </dd>
        <dt>Height</dt>
        <dd>{camera.height.toFixed(2)} m</dd>
        <dt>Heading / Pitch</dt>
        <dd>
          {camera.heading.toFixed(1)}° / {camera.pitch.toFixed(1)}°
        </dd>
        <dt>Ground / px</dt>
        <dd>
          {Number.isFinite(camera.metersPerPixel) ? `${camera.metersPerPixel.toFixed(3)} m` : "—"}
        </dd>
        <dt>Pending requests</dt>
        <dd>{pending || perf.pendingRequests}</dd>
        <dt>Tiles processing</dt>
        <dd>{perf.tilesProcessing}</dd>
        <dt>Tileset memory</dt>
        <dd>
          {memory} MB{perf.memoryBudgetMb ? ` / ${perf.memoryBudgetMb} MB` : ""}
        </dd>
        <dt>Site SSE</dt>
        <dd>{perf.siteScreenSpaceError ?? "—"}</dd>
        <dt>World SSE</dt>
        <dd>{perf.worldScreenSpaceError ?? "—"}</dd>
        <dt>Adaptive</dt>
        <dd>{perf.adaptiveReason}</dd>
        <dt>Resolution scale</dt>
        <dd>
          {perf.resolutionScale.toFixed(2)} × DPR {perf.devicePixelRatio.toFixed(2)}
        </dd>
        <dt>MSAA</dt>
        <dd>{perf.msaaSamples}×</dd>
        <dt>Profile</dt>
        <dd>{perf.profile}</dd>
        <dt>Motion fps</dt>
        <dd data-testid="dev-benchmark">
          {perf.benchmark.motionFps === null
            ? "move the camera"
            : `${perf.benchmark.motionFps.toFixed(0)} avg · p95 ${perf.benchmark.p95FrameMs?.toFixed(0)} ms · ${perf.benchmark.samples} frames`}
        </dd>
        <dt>Frame CPU</dt>
        <dd data-testid="dev-frame-budget">
          {perf.frameBudget.updateMs === null
            ? perf.frameBudget.loadingUpdateMs === null
              ? "—"
              : `loading: update ${perf.frameBudget.loadingUpdateMs.toFixed(1)} ms (${perf.frameBudget.loadingFrames} frames)`
            : `update ${perf.frameBudget.updateMs.toFixed(1)} ms · render ${perf.frameBudget.renderMs?.toFixed(1)} ms · ${perf.frameBudget.commands} draws` +
              (perf.frameBudget.loadingUpdateMs === null
                ? ""
                : ` · while loading ${perf.frameBudget.loadingUpdateMs.toFixed(0)} ms`)}
        </dd>
        <dt>Representation</dt>
        <dd>{representation ? representationLabel(representation) : "—"}</dd>
        <dt>Last API call</dt>
        <dd>{apiMs === null ? "—" : `${apiMs} ms`}</dd>
        <dt>WebGL 2</dt>
        <dd>{perf.webgl2 ? "yes" : "no"}</dd>
        <dt>GPU</dt>
        <dd style={{ overflowWrap: "anywhere" }}>{perf.gpu ?? "unavailable"}</dd>
      </dl>
      <Divider />
      <p className="glass-eyebrow">Active tilesets</p>
      {tilesets.length === 0 ? (
        <p className="glass-subtle" style={{ fontSize: "var(--text-xs)" }}>
          none
        </p>
      ) : (
        <ul className="glass-list" style={{ fontSize: "var(--text-xs)", gap: "0.15rem" }}>
          {tilesets.map((t) => (
            <li key={t} className="glass-mono">
              {t}
            </li>
          ))}
        </ul>
      )}
      <Divider />
      <Flag
        id="boundingVolumes"
        label="Tile bounding volumes"
        checked={flags.boundingVolumes}
        onChange={(v) => setFlag({ boundingVolumes: v })}
      />
      <Flag
        id="wireframe"
        label="Wireframe"
        checked={flags.wireframe}
        onChange={(v) => setFlag({ wireframe: v })}
      />
      <Flag
        id="tileCoordinates"
        label="Globe tile coordinates"
        checked={flags.tileCoordinates}
        onChange={(v) => setFlag({ tileCoordinates: v })}
      />
      <Flag
        id="fpsOverlay"
        label="Cesium FPS overlay"
        checked={flags.fpsOverlay}
        onChange={(v) => setFlag({ fpsOverlay: v })}
      />
      <Flag
        id="clipping"
        label="World clipping"
        checked={flags.clipping}
        onChange={(v) => setFlag({ clipping: v })}
      />
    </FloatingPanel>
  );
}
