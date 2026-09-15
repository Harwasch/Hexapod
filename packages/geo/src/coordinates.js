/** Coordinate formatting and simple spherical geodesy (WGS 84 mean radius). */
export const EARTH_RADIUS_M = 6_371_008.8;
export const DEG_TO_RAD = Math.PI / 180;
export const RAD_TO_DEG = 180 / Math.PI;
export function formatLatLon(position, digits = 5) {
  const lat = `${Math.abs(position.latitude).toFixed(digits)}° ${position.latitude >= 0 ? "N" : "S"}`;
  const lon = `${Math.abs(position.longitude).toFixed(digits)}° ${position.longitude >= 0 ? "E" : "W"}`;
  return `${lat}, ${lon}`;
}
export function formatDecimal(position, digits = 6) {
  return `${position.latitude.toFixed(digits)}, ${position.longitude.toFixed(digits)}`;
}
/** Great-circle distance in metres (haversine). */
export function haversineDistance(a, b) {
  const dLat = (b.latitude - a.latitude) * DEG_TO_RAD;
  const dLon = (b.longitude - a.longitude) * DEG_TO_RAD;
  const lat1 = a.latitude * DEG_TO_RAD;
  const lat2 = b.latitude * DEG_TO_RAD;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
  return 2 * EARTH_RADIUS_M * Math.asin(Math.min(1, Math.sqrt(h)));
}
/** Initial bearing from a to b in degrees [0, 360). */
export function bearing(a, b) {
  const lat1 = a.latitude * DEG_TO_RAD;
  const lat2 = b.latitude * DEG_TO_RAD;
  const dLon = (b.longitude - a.longitude) * DEG_TO_RAD;
  const y = Math.sin(dLon) * Math.cos(lat2);
  const x = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLon);
  return (((Math.atan2(y, x) * RAD_TO_DEG) % 360) + 360) % 360;
}
/** Point reached by travelling `distanceM` along `bearingDeg` from `origin`. */
export function destination(origin, bearingDeg, distanceM) {
  const delta = distanceM / EARTH_RADIUS_M;
  const theta = bearingDeg * DEG_TO_RAD;
  const lat1 = origin.latitude * DEG_TO_RAD;
  const lon1 = origin.longitude * DEG_TO_RAD;
  const lat2 = Math.asin(
    Math.sin(lat1) * Math.cos(delta) + Math.cos(lat1) * Math.sin(delta) * Math.cos(theta),
  );
  const lon2 =
    lon1 +
    Math.atan2(
      Math.sin(theta) * Math.sin(delta) * Math.cos(lat1),
      Math.cos(delta) - Math.sin(lat1) * Math.sin(lat2),
    );
  return { longitude: normalizeLongitude(lon2 * RAD_TO_DEG), latitude: lat2 * RAD_TO_DEG };
}
export function normalizeLongitude(longitude) {
  return ((((longitude + 180) % 360) + 360) % 360) - 180;
}
export function clampLatitude(latitude) {
  return Math.max(-90, Math.min(90, latitude));
}
