/**
 * Mission Control, built live in the page (landing.html #app): the app's Plans and Fleet panels
 * and its map's zone chips as real HTML over the app's view of the demo ranch, moving with
 * simulated data. Harrison picked this (2026-10-27) over the three screenshots: it stays sharp
 * on every screen, it moves, and a visitor can approve a plan and watch the fleet take it up.
 *
 * Everything shown is simulated and says so. The simulation (`makeSim`) is pure state stepped
 * by time, so the tests can run it without a page.
 */

export type PlanState = "draft" | "behind" | "on" | "running" | "queued" | "done";

export interface Plan {
  id: string;
  title: string;
  state: PlanState;
  /** 0 to 100. Plans counted in runs show `runs` instead. */
  progress: number;
  runs?: [done: number, of: number];
  meta: string;
  end: string;
  /** Percent per simulated minute. */
  rate: number;
}

export type MachineState = "work" | "warn" | "idle";

export interface Machine {
  id: string;
  name: string;
  kind: string;
  task: string;
  zone: string;
  state: MachineState;
  /** Battery, percent. */
  batt: number;
  /** Hours on shift, in simulated minutes. */
  mins: number;
}

export interface Sim {
  /** Simulated minutes since the page opened. */
  clock: number;
  plans: Plan[];
  machines: Machine[];
  agent: string;
  notice: string;
}

/** One second on the page is a simulated minute: a time-lapse, labelled as such. */
export const MINUTES_PER_SECOND = 1;

const AGENT = [
  "Sequencing passes against the 48 h wind forecast",
  "Scan of Z-08 finished: 214 broom plants flagged",
  "Holding seed drilling until 6 mm of rain falls",
  "Booked HX-12 a battery swap at the hub, 14:20",
  "Moved QD-18 up the ridge for a clearer radio path",
];

/** The broom plan the agent proposes; the visitor approves it. */
export const DRAFT_ID = "broom";

export function makeSim(): Sim {
  return {
    clock: 0,
    agent: AGENT[0] ?? "",
    notice:
      "CR-02 track wear is at 86%. The agent booked service for Thursday and moved its runs to CR-05.",
    plans: [
      {
        id: DRAFT_ID,
        title: "Pull French broom on the oak hillside",
        state: "draft",
        progress: 0,
        meta: "Proposed by the agent · Z-03 · 12 acres · 3 hexapods",
        end: "About 2 days",
        rate: 1.6,
      },
      {
        id: "thistle",
        title: "Pull yellow star thistle on the west grassland",
        state: "behind",
        progress: 54,
        meta: "Treatment · Z-14, Z-21 · 448 acres",
        end: "Ends Nov 20, 2026",
        rate: 0.03,
      },
      {
        id: "arundo",
        title: "Clear Arundo and blackberry from the creek draw",
        state: "on",
        progress: 82,
        runs: [9, 11],
        meta: "Recurring · 1.9 mi of Figueroa Creek · monthly",
        end: "Ongoing",
        rate: 0,
      },
      {
        id: "survey",
        title: "Collect a detailed 3D survey of the ranch",
        state: "running",
        progress: 62,
        meta: "One-off · 1,349 acres · lidar and photos",
        end: "Ends Dec 4, 2026",
        rate: 0.06,
      },
      {
        id: "seed",
        title: "Drill native seed after the first rain",
        state: "queued",
        progress: 0,
        meta: "Seasonal window · 448 acres",
        end: "Nov 12 to Dec 5",
        rate: 0,
      },
      {
        id: "fence",
        title: "Inspect the fence line and keep its fuel break clear",
        state: "on",
        progress: 74,
        meta: "Recurring · 2.9 mi · quarterly",
        end: "Ongoing",
        rate: 0.02,
      },
    ],
    machines: [
      m("HX-04", "Kestrel", "Hexapod", "Pulling thistle", "Z-14 west grassland", "work", 82, 374),
      m("HX-12", "Pipit", "Hexapod", "Pulling thistle", "Z-21 fence line", "work", 67, 242),
      m("CR-02", "Harrier", "Carrier", "Service due", "Hub", "warn", 41, 160),
      m("QD-18", "Shrike", "Quadcopter", "Low signal", "Z-08 creek draw", "warn", 54, 331),
      m("VT-01", "Vireo", "VTOL", "Docked", "Hub", "idle", 96, 0),
      m("CR-05", "Towhee", "Carrier", "Loading seed", "Hub", "idle", 88, 78),
    ],
  };
}

function m(
  id: string,
  name: string,
  kind: string,
  task: string,
  zone: string,
  state: MachineState,
  batt: number,
  mins: number,
): Machine {
  return { id, name, kind, task, zone, state, batt, mins };
}

/** Advances the simulation by `seconds` of page time. */
export function step(sim: Sim, seconds: number): Sim {
  const dm = seconds * MINUTES_PER_SECOND;
  const clock = sim.clock + dm;
  const plans = sim.plans.map((p) => {
    if (p.state === "draft" || p.state === "queued" || p.state === "done") return p;
    const progress = Math.min(100, p.progress + p.rate * dm);
    // An approved plan finishes; the long-running ones keep going.
    if (progress >= 100 && p.id === DRAFT_ID)
      return { ...p, progress: 100, state: "done" as const };
    return { ...p, progress: Math.min(99, progress) };
  });
  const machines = sim.machines.map((mc) => {
    let { state, task, zone, batt } = mc;
    // QD-18's radio drops out early on, then the agent moves it and it carries on.
    if (mc.id === "QD-18" && state === "warn" && clock > 8) {
      state = "work";
      task = "Survey sweep";
      zone = "Z-08 ridge";
    }
    if (state === "work") batt = Math.max(12, batt - dm * 0.08);
    else if (state === "idle" && zone === "Hub") batt = Math.min(100, batt + dm * 0.2);
    return {
      ...mc,
      state,
      task,
      zone,
      batt,
      mins: state === "work" ? mc.mins + dm : mc.mins,
    };
  });
  const turn = Math.floor(clock / 6) % AGENT.length;
  return { ...sim, clock, plans, machines, agent: AGENT[turn] ?? sim.agent };
}

/** The visitor approves the agent's plan: it starts, and three hexapods take it up. */
export function approve(sim: Sim): Sim {
  const plan = sim.plans.find((p) => p.id === DRAFT_ID);
  if (plan?.state !== "draft") return sim;
  return {
    ...sim,
    agent: "Plan approved. HX-07, HX-09 and HX-15 are walking to Z-03",
    plans: sim.plans.map((p) =>
      p.id === DRAFT_ID ? { ...p, state: "running", end: "Started just now" } : p,
    ),
    machines: [
      ...["HX-07 Wren", "HX-09 Junco", "HX-15 Phoebe"].map((label) => {
        const [id = "", name = ""] = label.split(" ");
        return m(id, name, "Hexapod", "Pulling broom", "Z-03 oak hillside", "work", 100, 0);
      }),
      ...sim.machines,
    ],
  };
}

/** The Fleet panel's four figures. */
export function figures(sim: Sim) {
  return {
    working: sim.machines.filter((mc) => mc.state === "work").length,
    attention: sim.machines.filter((mc) => mc.state === "warn").length,
  };
}

const STATE_TEXT: Record<PlanState, string> = {
  draft: "Needs approval",
  behind: "Behind",
  on: "On track",
  running: "Running",
  queued: "Queued",
  done: "Done",
};

const hm = (mins: number) =>
  `${Math.floor(mins / 60)}:${String(Math.floor(mins % 60)).padStart(2, "0")}`;

function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined) e.textContent = text;
  return e;
}

/**
 * Starts the live panels inside `root` (landing.html `#mc`). `buttons` switch its view: the
 * app's own Map, Plan and Fleet tabs, and the section's See, Plan and Do points.
 * `still`: reduced motion, which holds the simulation where it starts (approving still works).
 */
export function mountMissionControl(
  root: HTMLElement,
  buttons: HTMLButtonElement[],
  still: MediaQueryList,
) {
  let sim = makeSim();
  const plansList = root.querySelector<HTMLElement>("[data-mc=plans]");
  const fleetBody = root.querySelector<HTMLElement>("[data-mc=fleet]");
  const working = root.querySelector<HTMLElement>("[data-mc=working]");
  const attention = root.querySelector<HTMLElement>("[data-mc=attention]");
  const barFleet = root.querySelector<HTMLElement>("[data-mc=bar-fleet]");
  const agent = root.querySelectorAll<HTMLElement>("[data-mc=agent]");
  const notice = root.querySelector<HTMLElement>("[data-mc=notice]");
  const zone14 = root.querySelector<HTMLElement>("[data-mc=z14]");
  const map = root.querySelector<SVGSVGElement>("svg");
  if (!plansList || !fleetBody) throw new Error("landing: Mission Control's panels are missing");

  // Plans: one card each, built once; ticking updates only their numbers.
  const planRows = new Map<string, { card: HTMLElement; bar: HTMLElement; value: HTMLElement }>();
  function buildPlans() {
    plansList?.replaceChildren();
    planRows.clear();
    for (const p of sim.plans) {
      const card = el("li", "mc-plan");
      card.dataset.state = p.state;
      const top = el("div", "mc-plan-top");
      const title = el("b", undefined, p.title);
      const state = el("span", "mc-state", STATE_TEXT[p.state]);
      top.append(title, state);
      const bar = el("i");
      const track = el("div", "mc-bar-track");
      track.append(bar);
      const value = el("span", "mc-num");
      const prog = el("div", "mc-prog");
      prog.append(track, value);
      const foot = el("div", "mc-plan-foot");
      foot.append(el("span", undefined, p.meta), el("span", undefined, p.end));
      card.append(top, prog, foot);
      if (p.state === "draft") {
        const ok = el("button", "mc-approve", "Approve plan");
        ok.type = "button";
        ok.addEventListener("click", () => {
          sim = approve(sim);
          buildPlans();
          buildFleet();
          paint();
        });
        card.append(ok);
      }
      planRows.set(p.id, { card, bar, value });
      plansList?.append(card);
    }
  }

  const fleetRows = new Map<
    string,
    Record<"row" | "task" | "zone" | "batt" | "hrs", HTMLElement>
  >();
  function buildFleet() {
    fleetBody?.replaceChildren();
    fleetRows.clear();
    for (const mc of sim.machines) {
      const row = el("tr");
      const who = el("td");
      who.append(el("b", undefined, `${mc.id} ${mc.name}`), el("span", undefined, mc.kind));
      const task = el("td", "mc-task");
      const zone = el("td");
      const batt = el("td", "mc-num");
      const hrs = el("td", "mc-num mc-hrs");
      row.append(who, task, zone, batt, hrs);
      fleetRows.set(mc.id, { row, task, zone, batt, hrs });
      fleetBody?.append(row);
    }
  }

  function paint() {
    for (const p of sim.plans) {
      const r = planRows.get(p.id);
      if (!r) continue;
      if (r.card.dataset.state !== p.state) {
        r.card.dataset.state = p.state;
        const state = r.card.querySelector(".mc-state");
        if (state) state.textContent = STATE_TEXT[p.state];
      }
      r.bar.style.width = `${p.progress.toFixed(1)}%`;
      r.value.textContent = p.runs
        ? `${p.runs[0]} of ${p.runs[1]} runs`
        : p.state === "draft" || p.state === "queued"
          ? "Not started"
          : `${Math.floor(p.progress)}%`;
    }
    for (const mc of sim.machines) {
      const r = fleetRows.get(mc.id);
      if (!r) continue;
      r.row.dataset.state = mc.state;
      r.task.textContent = mc.task;
      r.zone.textContent = mc.zone;
      r.batt.textContent = `${Math.round(mc.batt)}%`;
      r.hrs.textContent = hm(mc.mins);
    }
    const f = figures(sim);
    if (working) working.textContent = String(f.working);
    if (attention) attention.textContent = String(f.attention);
    if (barFleet)
      barFleet.textContent = `Fleet: ${f.working} working · ${f.attention} need${f.attention === 1 ? "s" : ""} attention`;
    for (const a of agent) a.textContent = sim.agent;
    if (notice) notice.textContent = sim.notice;
    const thistle = sim.plans.find((p) => p.id === "thistle");
    if (zone14 && thistle) zone14.textContent = `pulling ${Math.floor(thistle.progress)}%`;
  }

  buildPlans();
  buildFleet();
  paint();

  // Runs while it is on screen, at one tick a second; the map's dots move by SVG animation.
  let timer = 0;
  let onScreen = false;
  const run = () => {
    const go = onScreen && !still.matches;
    if (go && !timer)
      timer = window.setInterval(() => {
        sim = step(sim, 1);
        paint();
      }, 1000);
    if (!go && timer) {
      clearInterval(timer);
      timer = 0;
    }
    if (map) {
      if (go) map.unpauseAnimations();
      else map.pauseAnimations();
    }
  };
  new IntersectionObserver((entries) => {
    onScreen = entries.some((e) => e.isIntersecting);
    run();
  }).observe(root);
  still.addEventListener("change", run);
  run();

  const panels = [...root.querySelectorAll<HTMLElement>("[data-for]")];
  function show(view: string) {
    root.dataset.view = view;
    for (const b of buttons) b.setAttribute("aria-pressed", String(b.dataset.view === view));
    for (const p of panels) p.hidden = p.dataset.for !== view;
  }
  for (const b of buttons) b.addEventListener("click", () => show(b.dataset.view ?? "see"));
}
