import { JulianDate } from "cesium";
import { useEffect } from "react";
import { useSite } from "@/api/queries";
import { useScene } from "@/cesium/SceneContext";
import { WEATHER_FLAG, weatherFeed } from "@/lib/weather/openMeteo";
import { useLivingLab } from "@/state/livingLab";
import { useInstances } from "@/state/instances";
import { useSites } from "@/state/sites";
import { setLabWeather, useLivingWeather } from "@/state/livingWeather";

function ScanVegetation() {
  const id = useSites((s) => s.activeSiteId),
    site = useSite(id).data;
  const assets = useInstances((s) => s.assets);
  useEffect(() => {
    const asset = site?.assets.find((a) => assets[a.id]);
    const source = asset?.source;
    const url =
      source?.type === "3d-tiles-url" ? new URL("instances.json", source.url).toString() : "";
    const lab = useLivingLab.getState();
    lab.set({ weatherSidecarUrl: url ? url + "?tier-a=1" : "" });
    if (lab.weather) setLabWeather(lab.weather);
  }, [site, assets]);
  return null;
}
/** Only mounted for live mode: follows the actual site's centroid and scene clock. */
function MapWeatherFeed() {
  const id = useSites((s) => s.activeSiteId),
    site = useSite(id).data,
    scene = useScene();
  const latitude = site?.centroid.latitude,
    longitude = site?.centroid.longitude,
    place = site?.name;
  useEffect(() => {
    let disposed = false,
      latest = "";
    const update = () => {
      if (!scene || latitude === undefined || longitude === undefined) {
        if (latest !== "missing") {
          latest = "missing";
          setLabWeather(null);
          useLivingWeather.getState().set({ note: "No weather, wind off · select a site" });
        }
        return;
      }
      const at = JulianDate.toDate(scene.viewer.clock.currentTime),
        hour = at.toISOString().slice(0, 13);
      if (hour === latest) return;
      latest = hour;
      setLabWeather(null);
      useLivingWeather.getState().set({ note: "No weather, wind off · loading weather…" });
      void weatherFeed(latitude, longitude, at).then((result) => {
        if (disposed || latest !== hour) return;
        setLabWeather(result.wind);
        useLivingWeather.getState().set({
          ...(result.wind ? { wind: result.wind } : {}),
          note: result.wind
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
    };
  }, [scene, latitude, longitude, place]);
  return null;
}
export function LivingWeatherControls() {
  const state = useLivingWeather();
  return (
    <fieldset>
      <legend>Simulated from weather</legend>
      <ScanVegetation />
      <label>
        Wind source{" "}
        <select
          aria-label="Wind source"
          value={state.mode}
          onChange={(e) => {
            const mode = e.target.value as "manual" | "live";
            state.set({ mode });
            if (mode === "manual") {
              setLabWeather(state.wind);
              state.set({ note: "Simulated · manual weather inputs" });
            } else setLabWeather(null);
          }}
        >
          <option value="manual">Manual</option>
          <option value="live">Live weather</option>
        </select>
      </label>
      {state.mode === "live" ? (
        <>
          <MapWeatherFeed />
          {!WEATHER_FLAG && <p>Weather feed disabled · VITE_LIVING_WEATHER=1 enables it.</p>}
        </>
      ) : (
        (
          [
            ["Mean at 10 m (m/s)", "meanMps", 0, 25, 0.1],
            ["Gust at 10 m (m/s)", "gustMps", 0, 40, 0.1],
            ["Wind from (°)", "directionDeg", 0, 360, 1],
          ] as const
        ).map(([label, key, min, max, step]) => (
          <label key={key} className="living-lab__row">
            {label} <output>{state.wind[key]?.toFixed(1)}</output>
            <input
              aria-label={label}
              type="range"
              min={min}
              max={max}
              step={step}
              value={state.wind[key] ?? 0}
              onChange={(e) => {
                const wind = { ...state.wind, [key]: Number(e.target.value) };
                wind.gustMps = Math.max(wind.meanMps, wind.gustMps ?? wind.meanMps);
                state.set({ wind });
                setLabWeather(wind);
              }}
            />
          </label>
        ))
      )}
      <p role="status">{state.note}</p>
    </fieldset>
  );
}
