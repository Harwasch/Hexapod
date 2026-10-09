import type { components } from "@twin/contracts";
type Request = components["schemas"]["EcologyRequest"];

export function ecologyNames(text: string, kingdom: string): Request["taxa"] | null {
  const names = text
    .split(/\r?\n/)
    .map((line) => line.trim().replace(/\s+/g, " "))
    .filter(Boolean);
  if (
    names.length > 20 ||
    names.some((name) => name.length > 200) ||
    new Set(names.map((name) => name.toLowerCase())).size !== names.length
  )
    return null;
  return names.map((scientificName) => ({ scientificName, kingdom: kingdom || null }));
}
