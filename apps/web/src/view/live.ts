/**
 * Live mode (`view.html#live/<captureId>`): a reconstruction as it happens.
 *
 * Polls `GET /captures/{id}/live` every few seconds and draws what the run has so far:
 * while the pose stage solves, each registered camera as a small frustum and a sample of
 * the sparse points in their own colours; while training runs, each intermediate splat
 * as it arrives, faded in over the last one, among the solved cameras. The person's view
 * is never reset by an update -- only the first thing drawn frames the camera.
 *
 * Everything the run shows here is in the pose solve's own frame (COLMAP's), not yet
 * placed east/north/up: it is turned so the solve's up estimate faces the sky, and centred
 * on the points. When the run finishes, the status panel links to the final scan.
 */
import { SparkRenderer, SplatFileType, SplatMesh } from "@sparkjsdev/spark";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import type { LiveCameras, LiveState, LiveTrainSeeds } from "@twin/contracts";

import { fetchApi } from "@/api/coldStart";
import { deviceSplatBudget } from "@/lib/detail";

import { liveConsole } from "./liveConsole";
import {
  decodeSeeds,
  distanceColour,
  isNewAttempt,
  nearestCameraDistance,
  snapshotAt,
} from "./liveTrain";

import {
  decodeCameras,
  decodePoints,
  describeLive,
  medianCentre,
  upOf,
  type DecodedCameras,
} from "./liveData";

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "";
const POLL_MS = 3_000;
const FADE_MS = 900;
const CAMERA_COLOUR = 0x7fd8c0;

function el(id: string): HTMLElement {
  const found = document.getElementById(id);
  if (!found) throw new Error(`#${id} is missing from view.html`);
  return found;
}

/** Frustum lines for every camera: apex to the four corners, and the far rectangle. */
function frustums(cameras: DecodedCameras, size: number, aspect: number): THREE.BufferGeometry {
  const positions = new Float32Array(cameras.count * 16 * 3);
  const depth = size;
  const halfH = size * 0.35;
  const halfW = halfH * (aspect > 0 ? aspect : 4 / 3);
  const c = new THREE.Vector3();
  const f = new THREE.Vector3();
  const u = new THREE.Vector3();
  const r = new THREE.Vector3();
  let at = 0;
  const push = (v: THREE.Vector3): void => {
    positions[at] = v.x;
    positions[at + 1] = v.y;
    positions[at + 2] = v.z;
    at += 3;
  };
  for (let i = 0; i < cameras.count; i += 1) {
    c.fromArray(cameras.centres, i * 3);
    f.fromArray(cameras.forward, i * 3);
    u.fromArray(cameras.up, i * 3);
    // COLMAP's camera x is right, y down, z forward: right = forward x up.
    r.crossVectors(f, u).normalize();
    const corner = (sx: number, sy: number): THREE.Vector3 =>
      c
        .clone()
        .addScaledVector(f, depth)
        .addScaledVector(r, sx * halfW)
        .addScaledVector(u, sy * halfH);
    const corners = [corner(-1, 1), corner(1, 1), corner(1, -1), corner(-1, -1)];
    for (const corner of corners) {
      push(c);
      push(corner);
    }
    for (let k = 0; k < 4; k += 1) {
      push(corners[k] ?? c);
      push(corners[(k + 1) % 4] ?? c);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  return geometry;
}

function srgbToLinear(values: Float32Array): Float32Array {
  return values.map((c) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)));
}

/** Median spacing between consecutive cameras: how big a frustum should be. */
function spacing(cameras: DecodedCameras, fallback: number): number {
  const gaps: number[] = [];
  for (let i = 1; i < cameras.count; i += 1) {
    const dx = (cameras.centres[i * 3] ?? 0) - (cameras.centres[i * 3 - 3] ?? 0);
    const dy = (cameras.centres[i * 3 + 1] ?? 0) - (cameras.centres[i * 3 - 2] ?? 0);
    const dz = (cameras.centres[i * 3 + 2] ?? 0) - (cameras.centres[i * 3 - 1] ?? 0);
    gaps.push(Math.hypot(dx, dy, dz));
  }
  gaps.sort((a, b) => a - b);
  const median = gaps[Math.floor(gaps.length / 2)] ?? 0;
  return median > 0 ? median : fallback;
}

export function showLive(captureId: string): { stop: () => void } {
  el("gallery").hidden = true;
  el("viewer").hidden = true;
  const section = el("live");
  section.hidden = false;
  section.dataset.cameras = "0";
  section.dataset.splatStep = "";
  const status = el("live-status");
  const detail = el("live-detail");
  const bar = el("live-bar");
  const track = el("live-track");
  const finalLink = el("live-final") as HTMLAnchorElement;
  finalLink.hidden = true;
  status.textContent = "Connecting…";
  detail.textContent = "";
  el("live-name").textContent = "Live";
  document.title = "Live";

  const renderer = new THREE.WebGLRenderer({ antialias: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0x0b0c0a);
  section.prepend(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(
    50,
    window.innerWidth / window.innerHeight,
    0.01,
    5000,
  );
  // This device's splat budget is Spark's here too, and centres stay 32-bit (see main.ts).
  const budget = deviceSplatBudget();
  scene.add(new SparkRenderer({ renderer, lodSplatCount: budget, accumExtSplats: true }));
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  controls.autoRotate = !reducedMotion;
  controls.autoRotateSpeed = 0.6;
  let touched = false;
  controls.addEventListener("start", () => {
    controls.autoRotate = false;
    touched = true;
  });

  // The solve's frame, turned so its up is three.js's +y and centred on the points.
  const frame = new THREE.Group();
  scene.add(frame);
  let cameraLines: THREE.LineSegments | null = null;
  let cloud: THREE.Points | null = null;
  let splat: { mesh: SplatMesh; name: string } | null = null;
  let loadingSplat: string | null = null;
  let framed = false;
  let lastSeq = -1;
  let camerasFinal = false;
  let layerMode: "splat" | "seeds" | "both" = "splat",
    camerasVisible = true;
  let seedPoints: THREE.Points<THREE.BufferGeometry, THREE.PointsMaterial> | null = null;
  let seedName: string | null = null,
    seedPending: string | null = null,
    seedEpoch = 0;
  const seedAbort = new AbortController();
  const disposeSeeds = (): void => {
    if (seedPoints) {
      frame.remove(seedPoints);
      seedPoints.geometry.dispose();
      seedPoints.material.dispose();
    }
    seedPoints = null;
    seedName = null;
    section.dataset.seeds = "0";
  };

  /**
   * A mid-solve snapshot is in the solve's frame of that moment; a splat is trained in the
   * finished model's, which COLMAP has re-centred and re-scaled since. Drawn together they
   * show the scene twice, at two sizes. So once a splat is up, only finished cameras stay.
   */
  const showOverlay = (): void => {
    const visible = splat === null || camerasFinal;
    if (cameraLines) cameraLines.visible = visible && camerasVisible;
    if (cloud) cloud.visible = visible && camerasVisible && layerMode === "splat";
    if (seedPoints) seedPoints.visible = layerMode !== "splat";
    if (splat) splat.mesh.visible = layerMode !== "seeds";
    for (const fade of fades) fade.mesh.visible = layerMode !== "seeds";
    section.dataset.layers = layerMode;
    section.dataset.camerasVisible = String(camerasVisible);
    section.dataset.seedsVisible = String(seedPoints?.visible ?? false);
  };
  let stopped = false;
  let timer = 0;
  const fades: { mesh: SplatMesh; from: number; to: number; start: number; done?: () => void }[] =
    [];

  let renderHost: HTMLElement = section;
  const resize = (): void => {
    const width = Math.max(1, renderHost.clientWidth),
      height = Math.max(1, renderHost.clientHeight);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
  };
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(section);
  renderer.setAnimationLoop((time: number) => {
    for (let i = fades.length - 1; i >= 0; i -= 1) {
      const fade = fades[i];
      if (!fade) continue;
      const t = reducedMotion ? 1 : Math.min(1, (time - fade.start) / FADE_MS);
      fade.mesh.opacity = fade.from + (fade.to - fade.from) * t;
      if (t >= 1) {
        fades.splice(i, 1);
        fade.done?.();
      }
    }
    controls.update();
    renderer.render(scene, camera);
  });

  /** Turn the solve's frame so `up` faces the sky, centred on `centre`. */
  const orient = (up: [number, number, number], centre: [number, number, number]): void => {
    frame.quaternion.setFromUnitVectors(new THREE.Vector3(...up), new THREE.Vector3(0, 1, 0));
    frame.position.copy(new THREE.Vector3(...centre).applyQuaternion(frame.quaternion).negate());
    frame.updateMatrixWorld(true);
  };

  /**
   * Fit the view to a sphere of `radius` around the origin. Again whenever the scene has
   * grown or shrunk by half since the last fit -- the solve starts with a handful of
   * cameras -- until the user takes the camera, after which it is theirs.
   */
  let framedRadius = 0;
  const frameView = (radius: number): void => {
    if (touched) return;
    if (framed && radius < framedRadius * 1.5 && radius > framedRadius / 1.5) return;
    framed = true;
    framedRadius = radius;
    const size = Math.max(radius, 1e-3);
    const halfV = THREE.MathUtils.degToRad(camera.fov / 2);
    const halfH = Math.atan(Math.tan(halfV) * camera.aspect);
    const distance = (size / Math.tan(Math.min(halfV, halfH))) * 1.1;
    controls.target.set(0, 0, 0);
    camera.position.copy(new THREE.Vector3(0, 0.5, 1).normalize().multiplyScalar(distance));
    camera.near = size / 1000;
    camera.far = size * 200;
    camera.updateProjectionMatrix();
    controls.minDistance = size * 0.02;
    controls.maxDistance = distance * 6;
  };

  /** The 90th-percentile distance of these points from `centre`: robust to strays. */
  const spreadRadius = (sets: Float32Array[], centre: readonly number[]): number | null => {
    const distances: number[] = [];
    for (const set of sets) {
      for (let i = 0; i + 2 < set.length; i += 3) {
        const dx = (set[i] ?? 0) - (centre[0] ?? 0);
        const dy = (set[i + 1] ?? 0) - (centre[1] ?? 0);
        const dz = (set[i + 2] ?? 0) - (centre[2] ?? 0);
        distances.push(Math.hypot(dx, dy, dz));
      }
    }
    if (distances.length < 3) return null;
    distances.sort((a, b) => a - b);
    return distances[Math.floor(distances.length * 0.9)] ?? null;
  };

  const drawCameras = (payload: LiveCameras, dimPoints: boolean): void => {
    const decoded = decodeCameras(payload);
    const points = decodePoints(payload);
    const centre = medianCentre(points.count ? points.positions : decoded.centres) ?? [
      payload.origin[0] ?? 0,
      payload.origin[1] ?? 0,
      payload.origin[2] ?? 0,
    ];
    // Until a splat is on screen the frame follows the solve, which re-normalises itself
    // as it grows; once one is, the frame holds still so the splat does not jump.
    if (!splat && !touched) orient(upOf({ cameras: payload, splat: null }), centre);
    cameraLines?.geometry.dispose();
    if (cameraLines) {
      (cameraLines.material as THREE.Material).dispose();
      frame.remove(cameraLines);
    }
    const size = spacing(decoded, payload.scale * 0.05) * 0.6;
    cameraLines = new THREE.LineSegments(
      frustums(decoded, size, payload.aspect ?? 4 / 3),
      new THREE.LineBasicMaterial({ color: CAMERA_COLOUR, transparent: true, opacity: 0.9 }),
    );
    frame.add(cameraLines);
    cloud?.geometry.dispose();
    if (cloud) {
      (cloud.material as THREE.Material).dispose();
      frame.remove(cloud);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(points.positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(srgbToLinear(points.colors), 3));
    cloud = new THREE.Points(
      geometry,
      new THREE.PointsMaterial({
        size: 2.5,
        sizeAttenuation: false,
        vertexColors: true,
        transparent: true,
        opacity: dimPoints ? 0.25 : 0.95,
      }),
    );
    frame.add(cloud);
    frame.updateMatrixWorld(true);
    section.dataset.cameras = String(decoded.count);
    section.dataset.points = String(points.count);
    // The solve's own spread, not `payload.scale` (the quantisation range, far larger).
    frameView(spreadRadius([points.positions, decoded.centres], centre) ?? payload.scale * 0.8);
    camerasFinal = payload.final;
    finalCentres = payload.final ? decoded.centres : null;
    showOverlay();
  };

  const showSplat = async (
    name: string,
    step: number,
    url: string,
    up: [number, number, number],
  ): Promise<void> => {
    const generation = ++loadGeneration;
    loadingSplat = name;
    const cacheKey = new URL(url, window.location.href).pathname;
    let request = byteCache.get(cacheKey);
    if (!request) {
      request = fetch(url)
        .then(async (response) => {
          if (!response.ok) throw new Error(`The splat answered ${String(response.status)}.`);
          return new Uint8Array(await response.arrayBuffer());
        })
        .catch((error: unknown) => {
          byteCache.delete(cacheKey);
          throw error;
        });
      byteCache.set(cacheKey, request);
    }
    const bytes = await request;
    if (stopped || desiredSplat !== name || generation !== loadGeneration) return;
    // A snapshot within the Detail budget is drawn whole, as it arrives; only a bigger one
    // waits for Spark to build its LoD tree (`lodAbove`), and is then drawn within it.
    const mesh = new SplatMesh({
      fileBytes: bytes,
      fileType: SplatFileType.SPZ,
      lod: true,
      lodAbove: budget,
      extSplats: true,
    });
    try {
      await mesh.initialized;
    } catch (error) {
      mesh.dispose();
      byteCache.delete(cacheKey);
      throw error;
    }
    if (stopped || desiredSplat !== name || generation !== loadGeneration) {
      mesh.dispose();
      return;
    }
    if (!cameraLines && !framed && !touched) {
      // No cameras to take the frame from: the splat's own centre and the pose up. A LoD
      // mesh keeps only its LoD tree, so the box is that tree's (its leaves are the splats).
      const lodSplats = mesh.packedSplats?.lodSplats ?? mesh.extSplats?.lodSplats;
      const box = lodSplats ? new THREE.Box3() : mesh.getBoundingBox(true);
      lodSplats?.forEachSplat((_index, centre) => box.expandByPoint(centre));
      const centre = box.getCenter(new THREE.Vector3());
      orient(up, [centre.x, centre.y, centre.z]);
      frameView(box.getSize(new THREE.Vector3()).length() / 2);
    }
    mesh.opacity = 0;
    frame.add(mesh);
    const now = performance.now();
    fades.push({ mesh, from: 0, to: 1, start: now });
    const previous = splat;
    if (previous) {
      for (let i = fades.length - 1; i >= 0; i -= 1) {
        if (fades[i]?.mesh === previous.mesh) fades.splice(i, 1);
      }
      fades.push({
        mesh: previous.mesh,
        from: previous.mesh.opacity,
        to: 0,
        start: now,
        done: () => {
          frame.remove(previous.mesh);
          previous.mesh.dispose();
        },
      });
    }
    splat = { mesh, name };
    if (cloud) (cloud.material as THREE.PointsMaterial).opacity = 0.25;
    showOverlay();
    section.dataset.splat = name;
    section.dataset.splatStep = String(step);
    consoleView.snapshot(step);
    void colourSplat();
  };

  const byteCache = new Map<string, Promise<Uint8Array>>();
  let desiredSplat: string | null = null;
  let loadGeneration = 0;
  let latestState: LiveState | null = null;
  let scrubStep: number | null = null;
  let finalCentres: Float32Array | null = null;
  let distanceMode = false;
  let colourGeneration = 0;
  const originals = new WeakMap<SplatMesh, [Uint32Array, Uint32Array]>();
  /** Spark 2.2.0 ExtSplats.getSplat/setSplat edit the backing texture arrays. Work in
   * short chunks; all final camera centres (already capped at 400) stay in solve space. */
  async function colourSplat(): Promise<void> {
    const generation = ++colourGeneration;
    const mesh = splat?.mesh,
      source = mesh?.extSplats;
    if (!mesh || !source) return;
    let original = originals.get(mesh);
    if (!original) {
      original = [source.extArrays[0].slice(), source.extArrays[1].slice()];
      originals.set(mesh, original);
    }
    source.extArrays[0].set(original[0]);
    source.extArrays[1].set(original[1]);
    const scale = latestState?.train?.plan?.metresPerUnit;
    const centres = finalCentres;
    mesh.maxSh = distanceMode ? 0 : 3;
    if (distanceMode && scale && centres) {
      for (let start = 0; start < source.numSplats; start += 256) {
        if (stopped || generation !== colourGeneration || mesh !== splat?.mesh) return;
        const end = Math.min(source.numSplats, start + 256);
        for (let i = start; i < end; i += 1) {
          const value = source.getSplat(i);
          const distance = nearestCameraDistance(
            value.center.x,
            value.center.y,
            value.center.z,
            centres,
          );
          if (distance != null) {
            value.color.setRGB(...distanceColour(distance * scale));
            source.setSplat(
              i,
              value.center,
              value.scales,
              value.quaternion,
              value.opacity,
              value.color,
            );
          }
        }
        await new Promise<void>((resolve) => window.setTimeout(resolve, 0));
      }
    }
    if (stopped || generation !== colourGeneration || mesh !== splat?.mesh) return;
    for (const texture of source.textures) texture.needsUpdate = true;
    mesh.updateGenerator();
    mesh.updateVersion();
  }
  const orbit = {
    position: camera.position.clone(),
    target: controls.target.clone(),
    up: camera.up.clone(),
  };
  let top = false;
  async function loadSeeds(payload: LiveTrainSeeds): Promise<void> {
    if (!payload.url || seedName === payload.key || seedPending) return;
    const epoch = seedEpoch;
    seedPending = payload.key;
    try {
      const response = await fetch(payload.url, { signal: seedAbort.signal });
      if (!response.ok) throw new Error("Seeds not available");
      const decoded = decodeSeeds(await response.arrayBuffer(), payload);
      if (stopped || seedEpoch !== epoch || latestState?.train?.seeds?.key !== payload.key) return;
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(decoded.positions, 3));
      geometry.setAttribute("color", new THREE.BufferAttribute(srgbToLinear(decoded.colors), 3));
      const material = new THREE.PointsMaterial({
        size: 2,
        sizeAttenuation: false,
        vertexColors: true,
      });
      disposeSeeds();
      seedPoints = new THREE.Points(geometry, material);
      seedName = payload.key;
      frame.add(seedPoints);
      section.dataset.seeds = String(payload.count);
      showOverlay();
    } catch {
      // A fresh URL on the next poll retries a checkpoint that has not synced yet.
    } finally {
      if (seedEpoch === epoch) seedPending = null;
    }
  }
  const consoleView = liveConsole({
    layers: (mode, cameras) => {
      layerMode = mode;
      camerasVisible = cameras;
      showOverlay();
    },
    scrub: (step) => {
      scrubStep = step;
      if (latestState) selectSnapshot(latestState);
    },
    color: (distance) => {
      distanceMode = distance;
      section.dataset.color = distance ? "distance" : "photo";
      void colourSplat();
    },
    view: (next) => {
      if (next === top) return;
      touched = true;
      controls.autoRotate = false;
      if (next) {
        orbit.position.copy(camera.position);
        orbit.target.copy(controls.target);
        orbit.up.copy(camera.up);
        const distance = camera.position.distanceTo(controls.target);
        camera.position.copy(controls.target).add(new THREE.Vector3(0, distance, 0));
        camera.up.set(0, 0, -1);
      } else {
        camera.position.copy(orbit.position);
        controls.target.copy(orbit.target);
        camera.up.copy(orbit.up);
      }
      top = next;
      camera.lookAt(controls.target);
      controls.update();
      section.dataset.view = next ? "top" : "orbit";
    },
  });
  function selectSnapshot(state: LiveState): void {
    const snapshots = state.train?.snapshots ?? [];
    const chosen = scrubStep == null ? null : snapshotAt(snapshots, scrubStep);
    // Before the first snapshot there is nothing to scrub to; retain the current view.
    if (scrubStep != null && !chosen) return;
    const newest = snapshots.find((snapshot) =>
      snapshot.key.endsWith(`/${state.splat?.name ?? ""}`),
    );
    const next = chosen
      ? { name: chosen.key, step: chosen.step, url: chosen.url }
      : state.splat
        ? { ...state.splat, name: newest?.key ?? state.splat.name }
        : null;
    desiredSplat = next?.name ?? null;
    if (next?.url && next.name !== splat?.name && next.name !== loadingSplat) {
      void showSplat(next.name, next.step, next.url, upOf(state))
        .catch(() => {
          // Also retry a failed scrub after the run has ended and ordinary polling stopped.
          if (!stopped) timer = window.setTimeout(() => void tick(), POLL_MS);
        })
        .finally(() => {
          if (loadingSplat === next.name) loadingSplat = null;
        });
    }
  }

  const render = (state: LiveState): boolean => {
    el("live-name").textContent = state.captureName;
    document.title = `Live · ${state.captureName}`;
    const described = describeLive(state);
    status.textContent = described.headline;
    detail.textContent = described.detail;
    track.hidden = described.fraction === null;
    bar.style.width = `${String(Math.round((described.fraction ?? 0) * 100))}%`;
    track.setAttribute("aria-valuenow", String(Math.round((described.fraction ?? 0) * 100)));
    if (state.cameras && state.cameras.seq !== lastSeq) {
      lastSeq = state.cameras.seq;
      drawCameras(state.cameras, splat !== null);
    }
    if (isNewAttempt(latestState?.train, state.train)) {
      seedEpoch += 1;
      seedPending = null;
      disposeSeeds();
      loadGeneration += 1;
      colourGeneration += 1;
      loadingSplat = null;
      byteCache.clear();
      const oldMeshes = new Set(fades.map((fade) => fade.mesh));
      if (splat) oldMeshes.add(splat.mesh);
      for (const mesh of oldMeshes) {
        frame.remove(mesh);
        mesh.dispose();
      }
      fades.length = 0;
      splat = null;
      section.dataset.splatStep = "";
    }
    latestState = state;
    if (consoleView.update(state) && renderHost === section) {
      renderHost = el("lc-scene");
      renderHost.prepend(renderer.domElement);
      resizeObserver.unobserve(section);
      resizeObserver.observe(renderHost);
      resize();
    }
    selectSnapshot(state);
    if (state.train?.seeds) void loadSeeds(state.train.seeds);
    if (described.ended && state.siteId) {
      finalLink.href = `#${state.siteId}`;
      finalLink.hidden = false;
    }
    section.dataset.status = state.status;
    return (
      described.ended &&
      (!state.splat || desiredSplat === splat?.name) &&
      (!state.train?.seeds || seedName === state.train.seeds.key)
    );
  };

  const tick = async (): Promise<void> => {
    window.clearTimeout(timer);
    if (stopped) return;
    let ended = false;
    try {
      const response = await fetchApi(
        `${API_BASE}/api/v1/captures/${encodeURIComponent(captureId)}/live`,
        { cache: "no-store" },
      );
      if (response.status === 404) {
        status.textContent = "Nothing to watch yet";
        detail.textContent = "This capture has no run. Start one from the phone page.";
      } else if (response.ok) {
        ended = render((await response.json()) as LiveState);
      }
    } catch {
      // A dropped request on a phone is normal; the next tick asks again.
    }
    if (!stopped && !ended) timer = window.setTimeout(() => void tick(), POLL_MS);
  };
  const onVisible = (): void => {
    if (document.visibilityState === "visible") void tick();
  };
  document.addEventListener("visibilitychange", onVisible);
  void tick();

  return {
    stop: () => {
      stopped = true;
      seedAbort.abort();
      seedEpoch += 1;
      disposeSeeds();
      window.clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisible);
      resizeObserver.disconnect();
      consoleView.stop();
      colourGeneration += 1;
      byteCache.clear();
      for (const mesh of new Set(fades.map((fade) => fade.mesh))) {
        if (mesh !== splat?.mesh) {
          frame.remove(mesh);
          mesh.dispose();
        }
      }
      cameraLines?.geometry.dispose();
      if (cameraLines) (cameraLines.material as THREE.Material).dispose();
      cloud?.geometry.dispose();
      if (cloud) (cloud.material as THREE.Material).dispose();
      renderer.setAnimationLoop(null);
      controls.dispose();
      splat?.mesh.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      section.hidden = true;
    },
  };
}
