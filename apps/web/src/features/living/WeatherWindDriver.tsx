import { JulianDate } from "cesium";
import { useEffect, useMemo, useRef, useState } from "react";

import { useSite } from "@/api/queries";
import { useScene } from "@/cesium/SceneContext";
import { weatherArticulationTier } from "@/cesium/scanView/livingArticulation";
import { weatherFeed } from "@/lib/weather/openMeteo";
import { probeRig } from "@/lib/weather/rigProbe";
import { planWeatherWind, type WindPlan, type WindScan } from "@/lib/weather/weatherWindPlan";
import { useAsCaptured } from "@/state/asCaptured";
import { useInstances } from "@/state/instances";
import { useLivingLab } from "@/state/livingLab";
import { setLabWeather } from "@/state/livingWeather";
import { useSettings, useSplatRenderer } from "@/state/settings";
import { useSites } from "@/state/sites";
import { useWeatherWind, weatherInput } from "@/state/weatherWind";

/** The site's hourly wind at the map's time, while live weather is wanted. */
function LiveWeatherFeed({
  latitude,
  longitude,
  place,
}: {
  latitude: number | undefined;
  longitude: number | undefined;
  place: string | undefined;
}) {
  const scene = useScene();
  useEffect(() => {
    const store = useWeatherWind.getState();
    let disposed = false,
      latest = "";
    const update = () => {
      if (!scene || latitude === undefined || longitude === undefined) {
        if (latest !== "missing") {
          latest = "missing";
          store.set({ live: null, feedNote: "No weather · select a site" });
        }
        return;
      }
      const at = JulianDate.toDate(scene.viewer.clock.currentTime),
        hour = at.toISOString().slice(0, 13);
      if (hour === latest) return;
      latest = hour;
      store.set({ live: null, feedNote: "Loading weather…" });
      void weatherFeed(latitude, longitude, at).then((result) => {
        if (disposed || latest !== hour) return;
        store.set({
          live: result.wind,
          feedNote: result.wind
            ? result.note.replace(
                `${latitude.toFixed(5)}, ${longitude.toFixed(5)}`,
                place ?? "site",
              )
            : result.note,
        });
      });
    };
    update();
    const remove = scene?.viewer.clock.onTick.addEventListener(update);
    return () => {
      disposed = true;
      remove?.();
      useWeatherWind.getState().set({ live: null, feedNote: "" });
    };
  }, [scene, latitude, longitude, place]);
  return null;
}

/** Strips the weather query, the articulation the renderer was asked for. */
function baseOf(url: string): string {
  try {
    const at = new URL(url, window.location.href);
    at.searchParams.delete("weather");
    at.searchParams.delete("breeze");
    return at.toString();
  } catch {
    return url;
  }
}

/**
 * Turns the weather wind switch into renderer state (the living hook, cesium/scanView/
 * livingHook.ts). Mounted only in a `VITE_LIVING_WEATHER` build. As captured and reduced motion
 * turn it fully off: no hook, no weather request. It writes the renderer's state only while it is
 * the one moving plants, so the developer lab (`?living=1`) keeps its own choices otherwise.
 */
export function WeatherWindDriver() {
  const enabled = useWeatherWind((s) => s.enabled);
  const mode = useWeatherWind((s) => s.mode);
  const manual = useWeatherWind((s) => s.manual);
  const live = useWeatherWind((s) => s.live);
  const captured = useAsCaptured((s) => s.on);
  const reduced = useSettings((s) => s.reducedMotion);
  const renderer = useSplatRenderer();
  const id = useSites((s) => s.activeSiteId);
  const site = useSite(id).data;
  const segmented = useInstances((s) => s.assets);
  const active = enabled && !captured && !reduced;

  const scans = useMemo<WindScan[]>(
    () =>
      (site?.assets ?? []).map((a) => ({
        assetId: a.id,
        name: a.name,
        tileset: a.source.type === "3d-tiles-url" ? a.source.url : "",
        splat: a.representation === "gaussian-splat",
        rigged: Boolean(a.renderConfig.rigUrl),
        segmented: segmented[a.id] !== undefined,
      })),
    [site, segmented],
  );

  const [rigFound, setRigFound] = useState<Record<string, boolean>>({});
  useEffect(() => {
    if (!active) return;
    let disposed = false;
    for (const scan of scans)
      if (scan.splat && !scan.rigged && scan.tileset)
        void probeRig(scan.assetId, scan.tileset).then((found) => {
          if (!disposed) setRigFound((was) => ({ ...was, [scan.assetId]: found }));
        });
    return () => {
      disposed = true;
    };
  }, [active, scans]);

  const plan: WindPlan | null = active ? planWeatherWind(scans, rigFound, renderer) : null;
  const latitude = site?.centroid.latitude,
    longitude = site?.centroid.longitude;
  const input = active
    ? weatherInput({
        mode,
        manual: {
          ...manual,
          latitude: latitude ?? manual.latitude,
          longitude: longitude ?? manual.longitude,
        },
        live,
      })
    : null;
  const planKey = JSON.stringify(plan);
  const inputKey = JSON.stringify(input);

  /** Whether this driver put the current wind on the renderer. */
  const driving = useRef(false);
  /** Articulations that loaded as "no vegetation": not asked for again this page. */
  const offUrls = useRef(new Set<string>());
  const release = () => {
    if (driving.current) setLabWeather(null);
    driving.current = false;
  };

  useEffect(() => {
    const store = useWeatherWind.getState();
    if (plan === null) {
      release();
      store.set({ scan: null, reason: "" });
      return;
    }
    if (plan.kind !== "target") {
      release();
      store.set({
        scan: null,
        reason: plan.kind === "pending" ? "Looking for vegetation data…" : plan.reason,
      });
      return;
    }
    const was = store.scan?.assetId === plan.assetId ? store.scan.tier : null;
    if (offUrls.current.has(plan.url)) {
      release();
      store.set({ scan: { assetId: plan.assetId, name: plan.name, tier: "off" }, reason: "" });
      return;
    }
    if (input === null) {
      release();
      store.set({ scan: { assetId: plan.assetId, name: plan.name, tier: was }, reason: "" });
      return;
    }
    setLabWeather(input, plan.url);
    driving.current = true;
    store.set({ scan: { assetId: plan.assetId, name: plan.name, tier: was }, reason: "" });
    // planKey and inputKey carry plan and input by value.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [planKey, inputKey]);

  // The tier is known once the renderer has loaded the articulation (its header's detail).
  useEffect(
    () =>
      useLivingLab.subscribe((lab, prev) => {
        if (!driving.current || lab.articulationStatus === prev.articulationStatus) return;
        const store = useWeatherWind.getState();
        const scan = store.scan;
        if (!scan || !lab.articulationUrl) return;
        const status = lab.articulationStatus;
        if (status.startsWith("Simulated")) {
          const tier = weatherArticulationTier(lab.articulationUrl) ?? null;
          store.set({ scan: { ...scan, tier } });
          if (tier === "off") {
            offUrls.current.add(baseOf(lab.articulationUrl));
            release();
          }
        } else if (!status.startsWith("Loading") && !status.startsWith("No articulation")) {
          release();
          store.set({ scan: { ...scan, tier: null }, reason: `wind off · ${status}` });
        }
      }),
    [],
  );

  // Unmounting (a build switch in tests, or the shell going away) leaves nothing moving.
  useEffect(() => () => release(), []);

  return active && mode === "live" ? (
    <LiveWeatherFeed latitude={latitude} longitude={longitude} place={site?.name} />
  ) : null;
}
