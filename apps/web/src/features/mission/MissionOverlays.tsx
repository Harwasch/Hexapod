import { useEffect, useRef } from "react";

import { useScene } from "@/cesium/SceneContext";
import type { Machine, Zone } from "@/missions/types";
import { useMission } from "@/state/mission";

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
 */
export function MissionOverlays() {
  const scene = useScene();
  const project = useMission((s) => s.project);
  const selection = useMission((s) => s.selection);
  const hovered = useMission((s) => s.hoveredMachineId);
  const setHovered = useMission((s) => s.setHoveredMachine);
  const zonesOn = useMission((s) => s.layers.zones);
  const { selectMachine, selectZone } = useMissionActions();
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
  }, [scene, project]);

  if (!project) return null;
  const register = (id: string) => (el: HTMLElement | null) => {
    if (el) nodes.current.set(id, el);
    else nodes.current.delete(id);
  };

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
