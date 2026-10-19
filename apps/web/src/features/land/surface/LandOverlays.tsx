import { useEffect, useMemo, useRef } from "react";

import { LandAnchorProjector } from "@/cesium/LandAnchorProjector";
import { useScene } from "@/cesium/SceneContext";
import { currentLeaders, onLeadersChange, useLandMap, type MapPin } from "@/state/landMap";

import { leaderPath } from "./leaderPath";

const SVG = "http://www.w3.org/2000/svg";

function leaderNode(svg: SVGSVGElement): SVGGElement {
  const group = document.createElementNS(SVG, "g");
  for (const className of ["land-leader__shadow", "land-leader__line"]) {
    const path = document.createElementNS(SVG, "path");
    path.setAttribute("class", className);
    group.appendChild(path);
  }
  for (const className of ["land-leader__start", "land-leader__ring", "land-leader__end"]) {
    const circle = document.createElementNS(SVG, "circle");
    circle.setAttribute("class", className);
    group.appendChild(circle);
  }
  svg.appendChild(group);
  return group;
}

/**
 * Land marks that live on the map: labelled pins and the leader lines that tie a card to the
 * place it describes. Both are positioned straight into the DOM after each rendered frame, as
 * the fleet markers are; React renders only when the set of pins changes.
 */
export function LandOverlays() {
  const scene = useScene();
  const pinsByOwner = useLandMap((s) => s.pins);
  const pins = useMemo(() => Object.values(pinsByOwner).flat(), [pinsByOwner]);
  const root = useRef<HTMLDivElement>(null);
  const svg = useRef<SVGSVGElement>(null);
  const nodes = useRef(new Map<string, HTMLElement>());
  const projector = useRef<LandAnchorProjector | null>(null);
  const shown = useRef(pins);

  useEffect(() => {
    if (!scene) return;
    const created = new LandAnchorProjector(scene);
    projector.current = created;
    const drawn = new Map<string, SVGGElement>();
    const draw = () => {
      const host = root.current;
      const layer = svg.current;
      if (!host || !layer || scene.isDestroyed) return;
      const origin = host.getBoundingClientRect();
      const canvas = scene.scene.canvas.getBoundingClientRect();
      const dx = canvas.left - origin.left;
      const dy = canvas.top - origin.top;
      for (const pin of shown.current) {
        const node = nodes.current.get(pin.id);
        if (!node) continue;
        const at = created.project(pin.position);
        node.style.visibility = at.visible ? "" : "hidden";
        if (at.visible)
          node.style.transform = `translate3d(${(at.x + dx).toFixed(1)}px, ${(at.y + dy).toFixed(1)}px, 0)`;
      }
      const live = new Set<string>();
      for (const leader of currentLeaders()) {
        const element = leader.element;
        if (!element.isConnected) continue;
        const rect = element.getBoundingClientRect();
        if (!rect.width || !rect.height) continue;
        const at = created.project(leader.anchor);
        const x = at.x + dx;
        const y = at.y + dy;
        const card = {
          left: rect.left - origin.left,
          top: rect.top - origin.top,
          right: rect.right - origin.left,
          bottom: rect.bottom - origin.top,
        };
        const path = at.visible ? leaderPath(card, x, y) : null;
        if (!path) continue;
        live.add(leader.id);
        let group = drawn.get(leader.id);
        if (!group) {
          group = leaderNode(layer);
          drawn.set(leader.id, group);
        }
        group.setAttribute("class", `land-leader land-leader--${leader.tone}`);
        const [shadow, line, start, ring, end] = Array.from(group.children);
        shadow?.setAttribute("d", path.d);
        line?.setAttribute("d", path.d);
        for (const [circle, cx, cy, r] of [
          [start, path.sx, path.sy, 3.5],
          [ring, x, y, 7],
          [end, x, y, 2.5],
        ] as const) {
          circle?.setAttribute("cx", cx.toFixed(1));
          circle?.setAttribute("cy", cy.toFixed(1));
          circle?.setAttribute("r", String(r));
        }
      }
      for (const [id, group] of drawn)
        if (!live.has(id)) {
          group.remove();
          drawn.delete(id);
        }
    };
    const offFrame = created.onFrame(draw);
    const offLeaders = onLeadersChange(() => created.requestFrame());
    const resize = () => created.requestFrame();
    window.addEventListener("resize", resize);
    // Cards move without the scene redrawing (a panel opening, text wrapping); follow them.
    const observer = new ResizeObserver(resize);
    if (root.current) observer.observe(root.current);
    return () => {
      offFrame();
      offLeaders();
      window.removeEventListener("resize", resize);
      observer.disconnect();
      drawn.forEach((group) => group.remove());
      created.destroy();
      projector.current = null;
    };
  }, [scene]);

  // New pins wait for a position; ask for a frame rather than for the next camera move.
  useEffect(() => {
    shown.current = pins;
    projector.current?.requestFrame();
  }, [pins]);

  return (
    <div ref={root} className="land-overlays" data-testid="land-overlays">
      <svg ref={svg} className="land-leaders" aria-hidden="true" />
      {pins.map((pin) => (
        <Pin
          key={pin.id}
          pin={pin}
          register={(element) => {
            if (element) nodes.current.set(pin.id, element);
            else nodes.current.delete(pin.id);
          }}
        />
      ))}
    </div>
  );
}

function Pin({ pin, register }: { pin: MapPin; register: (element: HTMLElement | null) => void }) {
  const tone = pin.tone ?? "accent";
  const body = (
    <>
      <span className="land-pin__dot" aria-hidden="true" />
      <span className="land-pin__label">{pin.label}</span>
    </>
  );
  return pin.onSelect ? (
    <button
      ref={register}
      type="button"
      className={`land-pin land-pin--${tone}`}
      style={{ visibility: "hidden" }}
      onClick={pin.onSelect}
      data-testid={`land-pin-${pin.id}`}
    >
      {body}
    </button>
  ) : (
    <span
      ref={register}
      className={`land-pin land-pin--${tone} land-pin--static`}
      style={{ visibility: "hidden" }}
      data-testid={`land-pin-${pin.id}`}
    >
      {body}
    </span>
  );
}
