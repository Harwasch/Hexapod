/**
 * "One job, start to finish" (landing.html #mission): one crew's whole mission, scrubbed by
 * scroll. The section is tall; its stage stays pinned while the reader scrolls through it, and
 * how far they are through it picks the moment of the mission the 3D scene draws (scene.js
 * startMissionScene): plan, deploy, scan, then operate. Harrison picked this (2026-10-27) from
 * the inspiration study: one camera, one journey, as Hut 8's site does.
 *
 * This module is the page around the scene: the four steps, the status line, the callout and the
 * scroll position. It works without the scene (no WebGL): the steps still follow the scroll.
 */
import type { MissionFrame } from "./scene.js";

export const STAGES = ["plan", "deploy", "scan", "operate"] as const;

/**
 * How far the reader is through a section that pins its stage, 0 to 1: 0 while its top is at or
 * below the top of the window, 1 once its bottom reaches the bottom of the window.
 */
export function progressThrough(top: number, height: number, viewport: number): number {
  const run = height - viewport;
  if (run <= 0) return 0;
  return Math.max(0, Math.min(1, -top / run));
}

/** The step (0 to 3) for a progress, each step an equal stretch of the scroll. */
export function stepAt(progress: number): number {
  return Math.min(STAGES.length - 1, Math.max(0, Math.floor(progress * STAGES.length)));
}

/** The scroll position (from the section's top) where a step begins, a little into it. */
export function scrollForStep(step: number, height: number, viewport: number): number {
  return ((step + 0.02) / STAGES.length) * Math.max(0, height - viewport);
}

export interface MissionPage {
  progress: () => number;
  onFrame: (frame: MissionFrame) => void;
  /** No scene: the section shows its steps as plain text, unpinned. */
  fallback: () => void;
}

export function mountMission(section: HTMLElement): MissionPage {
  const steps = [...section.querySelectorAll<HTMLElement>("[data-stage]")];
  const stageName = section.querySelector<HTMLElement>("[data-ms=stage]");
  const facts = section.querySelector<HTMLElement>("[data-ms=facts]");
  const rail = section.querySelector<HTMLElement>("[data-ms=rail]");
  const callout = section.querySelector<HTMLElement>(".mission-callout");
  const calloutText = callout?.querySelector("span");
  let current = -1;

  const progress = () => {
    const r = section.getBoundingClientRect();
    return progressThrough(r.top, r.height, window.innerHeight);
  };

  function showStep(i: number) {
    if (i === current) return;
    current = i;
    section.dataset.stage = STAGES[i];
    steps.forEach((li, k) => {
      if (k === i) li.setAttribute("aria-current", "step");
      else li.removeAttribute("aria-current");
    });
    const li = steps[i];
    if (stageName && li)
      stageName.textContent = `${li.querySelector("em")?.textContent ?? ""} ${li.querySelector("b")?.textContent ?? ""}`;
  }

  // The steps follow the scroll on their own, so they work before (or without) the scene.
  const follow = () => {
    const p = progress();
    showStep(stepAt(p));
    if (rail) rail.style.transform = `scaleX(${p.toFixed(4)})`;
  };
  addEventListener("scroll", follow, { passive: true });
  addEventListener("resize", follow);
  follow();

  // A step's name jumps to the start of that step.
  steps.forEach((li, k) =>
    li.querySelector("button")?.addEventListener("click", () => {
      const top = section.getBoundingClientRect().top + window.scrollY;
      window.scrollTo({
        top: top + scrollForStep(k, section.offsetHeight, window.innerHeight),
        behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
      });
    }),
  );

  function onFrame(f: MissionFrame) {
    showStep(f.index);
    if (facts) {
      facts.replaceChildren(
        ...f.status.map(([label, value]) => {
          const s = document.createElement("span");
          const l = document.createElement("i");
          l.textContent = label;
          s.append(l, ` ${value}`);
          return s;
        }),
      );
    }
    if (!callout || !calloutText) return;
    const c = f.callout;
    const inside = c && c.xy[0] > 24 && c.xy[0] < c.W - 24 && c.xy[1] > 70 && c.xy[1] < c.H - 60;
    if (c && inside) {
      if (calloutText.textContent !== c.text) calloutText.textContent = c.text;
      callout.style.transform = `translate(${c.xy[0].toFixed(1)}px,${c.xy[1].toFixed(1)}px)`;
      callout.dataset.side = c.xy[0] > c.W * 0.7 ? "left" : "right";
      callout.style.opacity = "1";
    } else callout.style.opacity = "0";
  }

  function fallback() {
    section.dataset.scene = "fallback";
    removeEventListener("scroll", follow);
    steps.forEach((li) => li.removeAttribute("aria-current"));
  }

  return { progress, onFrame, fallback };
}
