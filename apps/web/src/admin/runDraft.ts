/**
 * The New-run form's draft, and what it becomes on the wire.
 *
 * Separated from the component because this is the part with a rule in it: a typed-in
 * string has to come back as the type the recipe's default had, and a field left alone
 * has to stay out of `jobs.params` entirely. Both are testable without rendering
 * anything, and both would be quietly wrong forever if they were not.
 */
import type { RecipeRead } from "@twin/contracts";

/** Overrides as the form holds them: stage id → parameter name → the typed-in text. */
export type Draft = Record<string, Record<string, string>>;

/**
 * A typed-in value, converted back to the type the recipe's default had.
 *
 * A stage that defaults `iterations: 30000` wants a number, and `{"iterations": "7000"}`
 * would reach `Recipe.with_params` as a string and quietly change what the stage does.
 * The default is the only type information there is, so it is what the text is coerced
 * to; anything else is sent as JSON where it parses and as a string where it does not.
 */
export function coerce(text: string, fallback: unknown): unknown {
  const trimmed = text.trim();
  if (typeof fallback === "number") {
    const value = Number(trimmed);
    return Number.isFinite(value) ? value : trimmed;
  }
  if (typeof fallback === "boolean") return trimmed === "true";
  if (typeof fallback === "string") return trimmed;
  try {
    return JSON.parse(trimmed) as unknown;
  } catch {
    return trimmed;
  }
}

/** The draft as `jobs.params`: only what was actually changed, keyed by stage id. */
export function overridesFrom(
  draft: Draft,
  recipe: RecipeRead | undefined,
): Record<string, Record<string, unknown>> {
  const params: Record<string, Record<string, unknown>> = {};
  for (const stage of recipe?.stages ?? []) {
    const edits = draft[stage.id];
    if (!edits) continue;
    const defaults = stage.params as Record<string, unknown>;
    for (const [name, text] of Object.entries(edits)) {
      const value = coerce(text, defaults[name]);
      // Unchanged fields are left out rather than sent back: `jobs.params` is meant to
      // record what this run asked for, and a copy of the recipe's own defaults would
      // make every run look parameterised and two runs look different when they are not.
      if (JSON.stringify(value) === JSON.stringify(defaults[name])) continue;
      params[stage.id] = { ...params[stage.id], [name]: value };
    }
  }
  return params;
}
