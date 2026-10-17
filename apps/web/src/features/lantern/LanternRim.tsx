import { useEffect, useRef } from "react";

import type { LanternFrame } from "@/cesium/lanternTypes";
import { useScene } from "@/cesium/SceneContext";

import "./lantern.css";
import { NOT_CHECKED } from "./useDatedSources";

/** Room around the ring for its glow and its two lines of text (px). */
const PAD = 36;
/** Average width of a rim glyph (px at the rim's type size), to fit text along an arc. */
const GLYPH_PX = 6.1;

/** The accuracy without the "not checked" clause, which the rim says on its own line. */
function stated(label: string): string {
  return label.replace(` · ${NOT_CHECKED}`, "");
}

/** Text along an arc: squeezed to the arc's length when it would run past it. */
function fit(element: SVGTextPathElement | null, text: string, arcLength: number): void {
  if (!element) return;
  if (element.textContent !== text) element.textContent = text;
  const room = arcLength * 0.94;
  if (text.length * GLYPH_PX > room) {
    element.setAttribute("textLength", String(Math.round(room)));
    element.setAttribute("lengthAdjust", "spacingAndGlyphs");
  } else {
    element.removeAttribute("textLength");
    element.removeAttribute("lengthAdjust");
  }
}

/**
 * The Time Lantern's rim: a ring with a soft glow around the lens (it hides the stippled edge
 * of the cut, worldShader.ts), what the lens shows on the top arc -- the source and its date --
 * and on the bottom arc its licence and stated accuracy, with "not checked against today's 3D
 * world" under it, always. A crosshair marks the point the labels describe; where the source has
 * no record there, the centre says so.
 *
 * It follows the lens frame by frame from `events.lantern`, moved by a transform inside an
 * animation frame: no React render per frame. Never takes the pointer, and is not a popover, so
 * the globe's render holds (uiActivity.ts) never throttle the lens. Hidden from assistive tech:
 * the strip's live line (TimeLanternControl) says the same in words.
 */
export function LanternRim() {
  const scene = useScene();
  const layer = useRef<HTMLDivElement>(null);
  const ring = useRef<SVGSVGElement>(null);
  const top = useRef<SVGTextPathElement>(null);
  const bottom = useRef<SVGTextPathElement>(null);
  const caption = useRef<SVGTextElement>(null);
  const centre = useRef<SVGTextElement>(null);

  useEffect(() => {
    if (!scene) return;
    let frameId = 0;
    let pending: LanternFrame | null = null;
    let radius = 0;
    const draw = () => {
      frameId = 0;
      const frame = pending;
      const host = layer.current;
      const svg = ring.current;
      if (!frame || !host || !svg) return;
      const lens = frame.mode === "off" ? null : frame.lens;
      const source = frame.source;
      host.dataset.visible = String(Boolean(lens && source));
      if (!lens || !source) return;
      if (lens.r !== radius) {
        radius = lens.r;
        const size = 2 * (radius + PAD);
        const c = radius + PAD;
        svg.setAttribute("width", String(size));
        svg.setAttribute("height", String(size));
        svg.setAttribute("viewBox", `0 0 ${size} ${size}`);
        for (const circle of svg.querySelectorAll<SVGCircleElement>("circle[data-ring]")) {
          circle.setAttribute("cx", String(c));
          circle.setAttribute("cy", String(c));
          circle.setAttribute("r", String(radius));
        }
        const rt = radius + 9;
        const rb = radius + 22;
        svg
          .querySelector("#lantern-arc-top")
          ?.setAttribute("d", `M ${c - rt} ${c} A ${rt} ${rt} 0 0 1 ${c + rt} ${c}`);
        svg
          .querySelector("#lantern-arc-bottom")
          ?.setAttribute("d", `M ${c - rb} ${c} A ${rb} ${rb} 0 0 0 ${c + rb} ${c}`);
        svg.querySelector("[data-cross]")?.setAttribute("transform", `translate(${c} ${c})`);
        centre.current?.setAttribute("x", String(c));
        centre.current?.setAttribute("y", String(c + 26));
        caption.current?.setAttribute("x", String(c));
        caption.current?.setAttribute("y", String(c + radius + 34));
      }
      // The canvas and the HUD share an origin; measured anyway, it costs two reads a frame.
      const canvas = scene.viewer.canvas.getBoundingClientRect();
      const own = host.getBoundingClientRect();
      const x = canvas.left - own.left + lens.x - radius - PAD;
      const y = canvas.top - own.top + lens.y - radius - PAD;
      svg.style.transform = `translate3d(${x}px, ${y}px, 0)`;
      const date = frame.part?.dateLabel ?? source.dateLabel;
      const accuracy = stated(frame.part?.accuracyLabel ?? source.accuracyLabel);
      fit(top.current, `${source.title} · ${date}`, Math.PI * (radius + 9));
      fit(bottom.current, `${source.license} · ${accuracy}`, Math.PI * (radius + 22));
      const note = frame.covered ? "" : `No ${source.sortDate.slice(0, 4)} record at this spot`;
      if (centre.current && centre.current.textContent !== note) centre.current.textContent = note;
    };
    const off = scene.events.on("lantern", (frame) => {
      pending = frame;
      if (!frameId) frameId = requestAnimationFrame(draw);
    });
    return () => {
      off();
      if (frameId) cancelAnimationFrame(frameId);
    };
  }, [scene]);

  return (
    <div
      ref={layer}
      className="lantern-rim"
      aria-hidden="true"
      data-visible="false"
      data-testid="lantern-rim"
    >
      <svg ref={ring} className="lantern-rim__ring" width="0" height="0">
        <defs>
          <path id="lantern-arc-top" d="" />
          <path id="lantern-arc-bottom" d="" />
          <filter id="lantern-glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="5" />
          </filter>
        </defs>
        <circle data-ring className="lantern-rim__glow" filter="url(#lantern-glow)" />
        <circle data-ring className="lantern-rim__line" />
        <g data-cross className="lantern-rim__cross">
          <line x1="-7" y1="0" x2="-2" y2="0" />
          <line x1="2" y1="0" x2="7" y2="0" />
          <line x1="0" y1="-7" x2="0" y2="-2" />
          <line x1="0" y1="2" x2="0" y2="7" />
        </g>
        <text className="lantern-rim__text lantern-rim__text--top" data-testid="lantern-rim-top">
          <textPath ref={top} href="#lantern-arc-top" startOffset="50%" textAnchor="middle" />
        </text>
        <text className="lantern-rim__text" data-testid="lantern-rim-bottom">
          <textPath ref={bottom} href="#lantern-arc-bottom" startOffset="50%" textAnchor="middle" />
        </text>
        <text ref={caption} className="lantern-rim__caption" textAnchor="middle">
          Not checked against today&apos;s 3D world
        </text>
        <text
          ref={centre}
          className="lantern-rim__centre"
          textAnchor="middle"
          data-testid="lantern-rim-centre"
        />
      </svg>
      <p className="lantern-rim__corner">
        Outside the lens: Google Photorealistic 3D Tiles (capture dates not published). Inside:
        dated imagery draped on the terrain, with today&apos;s 3D removed.
      </p>
    </div>
  );
}
