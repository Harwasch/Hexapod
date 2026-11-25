import schema from "../../../ios/Modules/Globe/Bridge/schema.json";
import type { BridgeMessage, BridgeSnapshot } from "./generated";

function object(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
function matches(type: string, value: unknown): boolean {
  if (type.endsWith("?"))
    return value === null || value === undefined || matches(type.slice(0, -1), value);
  if (type.endsWith("[]"))
    return Array.isArray(value) && value.every((v) => matches(type.slice(0, -2), v));
  if (type === "String") return typeof value === "string" && value.length > 0;
  if (type === "Double") return typeof value === "number" && Number.isFinite(value);
  const record = (schema.records as Record<string, Record<string, string>>)[type];
  if (
    !record ||
    !object(value) ||
    !Object.entries(record).every(([key, t]) => matches(t, value[key]))
  )
    return false;
  return (
    type !== "BridgeGeo" ||
    (Math.abs(value.lat as number) <= 90 && Math.abs(value.lon as number) <= 180)
  );
}
export function decodeMessage(value: unknown): BridgeMessage {
  if (!object(value) || value.version !== schema.version || typeof value.type !== "string")
    throw Error("Invalid bridge envelope");
  if (!Object.hasOwn(schema.messages, value.type)) throw Error("Unknown bridge message");
  const fields = (schema.messages as Record<string, Record<string, string>>)[value.type];
  if (!fields || !Object.entries(fields).every(([key, type]) => matches(type, value[key])))
    throw Error("Invalid bridge payload");
  return value as unknown as BridgeMessage;
}
export interface GlobeCommands {
  snapshot(value: BridgeSnapshot): void;
  selectScan(id: string): void;
  flyToScan(id: string): void;
  memoryWarning(): void;
}
/** Native commands only. JS events cannot be reflected into the command channel. */
export function dispatchMessage(value: unknown, commands: GlobeCommands): void {
  const message = decodeMessage(value);
  switch (message.type) {
    case "memoryWarning":
      commands.memoryWarning();
      break;
    case "snapshot":
      commands.snapshot(message.snapshot);
      break;
    case "selectScan":
      commands.selectScan(message.scanId);
      break;
    case "flyToScan":
      commands.flyToScan(message.scanId);
      break;
    default:
      throw Error("Expected native command");
  }
}
