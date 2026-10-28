/** Weather inputs are measured/modelled m/s; the resulting plant motion is Simulated.
 * This independent pure model leaves the shipped dimensionless WindSettings unchanged.
 * Sources and all judgement parameters: docs/LIVING_WEATHER_WIND.md.
 */
export interface WeatherWind {
  meanMps: number;
  gustMps?: number;
  gustFactor?: number;
  /** Meteorological FROM, clockwise from north (Open-Meteo convention). */
  directionDeg: number;
  latitude: number;
  longitude: number;
}
export interface WindField {
  meanMps: number;
  gustMps: number;
  east: number;
  north: number;
  alongM: number;
  crossM: number;
  phase: number;
}
const TAU = 2 * Math.PI;
const finite = (n: number, fallback = 0) => (Number.isFinite(n) ? n : fallback);
export function prepareWeatherWind(w: WeatherWind): WindField {
  const meanMps = Math.max(0, Math.min(60, finite(w.meanMps)));
  const gustMps = Math.max(
    meanMps,
    Math.min(90, finite(w.gustMps ?? meanMps * (w.gustFactor ?? 1))),
  );
  const bearing = ((finite(w.directionDeg) + 180) * Math.PI) / 180;
  const gustiness = (gustMps - meanMps) / Math.max(1, meanMps);
  // 20 s organized forcing at the reference U=4, G/U=1.5. Scaling is a
  // bounded visual inference, not a measured universal length-vs-gust law.
  const alongM = Math.max(20, Math.min(160, (40 + 20 * meanMps) / (1 + gustiness)));
  return {
    meanMps,
    gustMps,
    east: Math.sin(bearing),
    north: Math.cos(bearing),
    alongM,
    crossM: alongM * 0.4,
    phase: finite(w.latitude) * 0.173 + finite(w.longitude) * 0.119,
  };
}
/** Frozen field: F(x + U*dt*downwind,t+dt) = F(x,t). Local ENU metres. */
export function sampleWeatherWind(f: WindField, eastM: number, northM: number, timeS: number) {
  const along = eastM * f.east + northM * f.north - f.meanMps * timeS;
  const cross = -eastM * f.north + northM * f.east;
  const a = (TAU * along) / f.alongM,
    b = (TAU * cross) / f.crossM;
  const wave =
    0.6 * Math.sin(a + 0.27 * Math.sin(b) + f.phase) +
    0.4 * Math.sin(1.618 * a - 0.43 * b + 1.7 + f.phase);
  const alongMps = Math.max(0, f.meanMps + (f.gustMps - f.meanMps) * wave);
  const crossMps = 0.25 * (f.gustMps - f.meanMps) * Math.sin(0.73 * a + b + 0.9 + f.phase);
  return {
    alongMps,
    crossMps,
    eastMps: f.east * alongMps - f.north * crossMps,
    northMps: f.north * alongMps + f.east * crossMps,
  };
}
/** Neutral roughness log profile; within a 6 m canopy add Cionco attenuation.
 * z0=.3 m (terrain III); alpha=2 is an assumed dense-canopy example.
 */
export function plantWindRatio(heightM: number, canopyM = 6): number {
  const z = Math.max(0, finite(heightM));
  return (
    (Math.log((z + 0.3) / 0.3) / Math.log(10.3 / 0.3)) *
    Math.exp(2 * Math.min(0, z / Math.max(0.3, canopyM) - 1))
  );
}
/** U² at low speed, ~U at high speed (Vogel exponent -1), bounded angle.
 * 4 m/s transition, .10 rad and load=28 are conservative visual choices.
 */
export function weatherAngle(speedMps: number): number {
  const u = Math.max(0, finite(speedMps));
  const load = (u * u) / Math.sqrt(1 + (u / 4) ** 2);
  return (0.1 * load) / (28 + load);
}
/** Two-axis angular response; exact root/trunk/ground pin, no integration. */
export function weatherBranch(
  f: WindField,
  x: number,
  y: number,
  height: number,
  t: number,
  compliance: number,
  frequencyHz: number,
  phase: number,
  pinned = false,
  canopyM = 6,
) {
  if (pinned || f.gustMps === 0 || height <= 0 || compliance <= 0) return [0, 0, 0] as const;
  const v = sampleWeatherWind(f, x, y, t),
    ratio = plantWindRatio(height, canopyM);
  const angle = weatherAngle(v.alongMps * ratio) * compliance;
  const oscillation = 1 + 0.22 * Math.sin(TAU * frequencyHz * t + phase);
  const side =
    angle * 0.14 * Math.sin(TAU * frequencyHz * 0.83 * t + phase + 1.3) +
    weatherAngle(Math.abs(v.crossMps) * ratio) * Math.sign(v.crossMps) * compliance;
  // Rotation about +y leans east; about -x leans north.
  return [
    -(f.north * angle * oscillation + f.east * side),
    f.east * angle * oscillation - f.north * side,
    0,
  ] as const;
}
