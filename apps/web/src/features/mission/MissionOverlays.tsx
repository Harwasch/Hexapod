import { useCallback, useEffect, useRef } from "react";

import { useScene } from "@/cesium/SceneContext";
import { collapsesToPin, pinCounts, pinLabel } from "@/missions/sitePin";
import type { Machine, Zone } from "@/missions/types";
import { useMission } from "@/state/mission";
import { useViewer } from "@/state/viewer";

import { useMissionActions } from "./useMissionActions";

const STATUS_CLASS: Record<Machine["status"], string> = {
  working: "is-working",
  attention: "is-attention",
  idle: "is-idle",
};

/**
 * Machine markers and zone chips as DOM elements positioned by the
 * MissionManager every frame (design: MACHINE MARKER, ZONE GEOMETRY chips).
 * Positions are written straight to the DOM; React never re-renders per frame.
 *
 * At globe scale (above `GLOBE_SCALE_ALTITUDE_M`) the whole project is a few pixels across,
 * so its markers and chips collapse into one site pin with the site's name and a count;
 * the pin flies to the site. React re-renders only when the camera crosses that height.
 */
export function MissionOverlays() {
  const scene = useScene();
  const project = useMission((s) => s.project);
  const selection = useMission((s) => s.selection);
  const hovered = useMission((s) => s.hoveredMachineId);
  const setHovered = useMission((s) => s.setHoveredMachine);
  const zonesOn = useMission((s) => s.layers.zones);
  const collapsed = useViewer((s) => collapsesToPin(s.camera.altitude));
  const { selectMachine, selectZone, flyToProject } = useMissionActions();
  const nodes = useRef(new Map<string, HTMLElement>());

  useEffect(() => {
    if (!scene) return;
    return scene.mission.onFrame((anchors) => {
      for (const anchor of anchors) {
        const node = nodes.current.get(anchor.id);
        if (!node) continue;
        if (!anchor.visible) {
          node.style.visibility = "hidden";
          continue;
        }
        node.style.visibility = "";
        node.style.transform = `translate3d(${anchor.x.toFixed(1)}px, ${anchor.y.toFixed(1)}px, 0)`;
      }
    });
    // Re-subscribing when the pin swaps in asks for a frame, so it is placed straight away.
  }, [scene, project, collapsed]);

  const register = useCallback(
    (id: string) => (el: HTMLElement | null) => {
      if (el) nodes.current.set(id, el);
      else nodes.current.delete(id);
    },
    [],
  );

  if (!project) return null;

  if (collapsed) {
    const counts = pinCounts(project, zonesOn);
    // The "Anywhere" project has nothing on the ground to stand for until an area is drawn.
    if (counts.total === 0) return null;
    const label = pinLabel(project.name, counts);
    return (
      <div className="mc-overlays" aria-label="Sites on map">
        <button
          ref={register(`site:${project.id}`)}
          type="button"
          className="mc-site-pin"
          style={{ visibility: "hidden" }}
          onClick={flyToProject}
          aria-label={`${label}. Fly to the site`}
          title={label}
          data-testid="site-pin"
        >
          <span className="mc-site-pin__dot" aria-hidden="true" />
          <span className="mc-site-pin__name">{project.name}</span>
          <span className="mc-site-pin__count mc-mono" aria-hidden="true">
            {counts.total}
          </span>
        </button>
      </div>
    );
  }

  return (
    <div className="mc-overlays" aria-label="Fleet on map">
      {zonesOn &&
        project.zones.map((zone: Zone) => {
          const selected = selection?.kind === "zone" && selection.id === zone.id;
          return (
            <button
              key={zone.id}
              ref={register(`zone:${zone.id}`)}
              type="button"
              className={`mc-chip mc-chip--${zone.tone} ${selected ? "is-selected" : ""}`}
              style={{ visibility: "hidden" }}
              onClick={() => selectZone(zone.id)}
              aria-label={`${zone.name}: ${zone.short}`}
              data-testid={`zone-chip-${zone.id}`}
            >
              <span className="mc-chip__dot" aria-hidden="true" />
              <span className="mc-mono mc-chip__id">{zone.id}</span>
              <span className="mc-chip__short">{zone.short}</span>
            </button>
          );
        })}
      {project.machines.map((machine: Machine) => {
        const selected = selection?.kind === "machine" && selection.id === machine.id;
        const labelOn = selected || hovered === machine.id;
        return (
          <button
            key={machine.id}
            ref={register(`machine:${machine.id}`)}
            type="button"
            className={`mc-marker ${STATUS_CLASS[machine.status]} ${selected ? "is-selected" : ""}`}
            style={{ visibility: "hidden" }}
            onClick={() => selectMachine(machine.id)}
            onMouseEnter={() => setHovered(machine.id)}
            onMouseLeave={() => setHovered(null)}
            onFocus={() => setHovered(machine.id)}
            onBlur={() => setHovered(null)}
            aria-label={`${machine.name}, ${machine.task}`}
            data-testid={`machine-marker-${machine.id}`}
          >
            {machine.status === "working" && (
              <span className="mc-marker__pulse" aria-hidden="true" />
            )}
            {selected && <span className="mc-marker__halo" aria-hidden="true" />}
            <span className="mc-marker__dot" aria-hidden="true" />
            {labelOn && (
              <span className="mc-marker__label">
                {machine.id} · {machine.task}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
