import { useEffect, useRef, useState } from "react";

import { useScene } from "@/cesium/SceneContext";
import type { LandPoint } from "@/state/land";
import { useLandContext } from "@/state/landContext";

import { outlineFromCorners } from "./plansMap";
import type { Area } from "./planModel";

const OWNER = "plans-draw";

/** What an outline being drawn is for: a step's area, or an area to avoid. */
export type DrawTarget = { kind: "step"; id: string } | { kind: "avoid" };

/**
 * Drawing an area on the map, corner by corner: each click on the ground adds a corner and
 * waits for the next, until the person finishes, cancels, or presses Escape (which pauses).
 * The same ground pick the other land tools use, held under one owner so nothing else picks
 * at the same time; leaving the lens cancels it.
 */
export function usePlanDrawing() {
  const scene = useScene();
  const picker = useLandContext((s) => s.pointPicker);
  const [target, setTarget] = useState<DrawTarget | null>(null);
  const [points, setPoints] = useState<LandPoint[]>([]);
  const [error, setError] = useState<string | null>(null);
  const ticket = useRef(0);

  const release = () => {
    if (useLandContext.getState().pointPicker === OWNER) {
      useLandContext.getState().setPointPicker(null);
      scene?.areas.cancelPick();
    }
  };

  useEffect(
    () => () => {
      ticket.current += 1;
      if (useLandContext.getState().pointPicker === OWNER) {
        useLandContext.getState().setPointPicker(null);
        scene?.areas.cancelPick();
      }
    },
    [scene],
  );

  const loop = async (current: number) => {
    if (!scene) return;
    while (current === ticket.current) {
      useLandContext.getState().setPointPicker(OWNER);
      const point = await scene.areas.pickGround();
      if (current !== ticket.current) return;
      if (!point) {
        // Escape: stop picking but keep the corners, so the outline can be resumed or finished.
        if (useLandContext.getState().pointPicker === OWNER)
          useLandContext.getState().setPointPicker(null);
        return;
      }
      setPoints((previous) => [...previous, [point.longitude, point.latitude]]);
    }
  };

  const start = (next: DrawTarget) => {
    const current = ++ticket.current;
    release();
    setTarget(next);
    setPoints([]);
    setError(null);
    void loop(current);
  };

  const resume = () => {
    if (!target) return;
    const current = ++ticket.current;
    setError(null);
    void loop(current);
  };

  const stop = () => {
    ticket.current += 1;
    release();
    setTarget(null);
    setPoints([]);
    setError(null);
  };

  /** Closes the outline; null (with an error shown) when it cannot be an area yet. */
  const finish = (): { target: DrawTarget; area: Area } | null => {
    if (!target) return null;
    const outline = outlineFromCorners(points);
    if ("error" in outline) {
      setError(outline.error);
      return null;
    }
    const done = { target, area: outline.area };
    stop();
    return done;
  };

  return {
    target,
    points,
    error,
    /** True while the map is waiting for the next corner. */
    picking: picker === OWNER,
    canDraw: Boolean(scene),
    start,
    resume,
    undo: () => setPoints((previous) => previous.slice(0, -1)),
    finish,
    cancel: stop,
  };
}

export type PlanDrawing = ReturnType<typeof usePlanDrawing>;
