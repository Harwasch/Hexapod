/**
 * New run: a capture, a recipe, any parameter overridden, a provider, a tier, a cost.
 *
 * The parameter editor is generated from the recipe catalogue rather than written out,
 * which is what makes it survive Phase B: when B3 changes `mask` from `none` to `robust`
 * in a recipe file, the field appears here with no frontend change. The console does not
 * validate the overrides either — `POST /captures/{id}/process` runs them through
 * `Recipe.with_params` (A6) and this shows whatever it says, so there is exactly one
 * implementation of what a recipe will accept.
 */
import { useMemo, useState } from "react";

import { Play } from "lucide-react";

import { GlassButton, GlassField, GlassInput, GlassSelect, GlassSheet, useFieldId } from "@twin/ui";

import { ApiError } from "@/api/error";

import { estimateRun } from "./estimate";
import { formatBytes } from "./format";
import { useCaptures, useLaunchRun, useRecipes } from "./queries";
import { overridesFrom, type Draft } from "./runDraft";

export function NewRun({
  captureId,
  onClose,
  onLaunched,
}: {
  captureId: string;
  onClose: () => void;
  onLaunched: () => void;
}) {
  const captures = useCaptures();
  const catalogue = useRecipes();
  const launch = useLaunchRun();

  const [capture, setCapture] = useState(captureId);
  const [recipeName, setRecipeName] = useState("");
  const [provider, setProvider] = useState("");
  const [tier, setTier] = useState("");
  const [draft, setDraft] = useState<Draft>({});

  const captureField = useFieldId("run-capture");
  const recipeField = useFieldId("run-recipe");
  const providerField = useFieldId("run-provider");
  const tierField = useFieldId("run-tier");

  const recipes = catalogue.data?.recipes ?? [];
  // Derived, not synchronised in an effect: "the chosen recipe, or the first one the
  // catalogue offered" is a fact about this render, and writing it into state on arrival
  // would render twice and lose the choice whenever the catalogue refetched.
  const chosen = recipeName || (recipes[0]?.name ?? "");
  const recipe = recipes.find((entry) => entry.name === chosen);
  const providers = catalogue.data?.providers ?? [];
  const chosenProvider = providers.find((entry) => entry.name === provider);

  const selected = (captures.data ?? []).find((entry) => entry.id === capture);
  const bytes = (selected?.files ?? []).reduce((sum, file) => sum + (file.bytes ?? 0), 0);

  const estimate = useMemo(
    () =>
      estimateRun({
        recipe,
        bytes,
        interruptible: chosenProvider?.interruptible ?? false,
      }),
    [recipe, bytes, chosenProvider],
  );

  const params = overridesFrom(draft, recipe);
  const changed = Object.keys(params).length;
  const problem =
    launch.error instanceof ApiError
      ? (launch.error.problem?.detail ?? launch.error.message)
      : launch.error
        ? String(launch.error)
        : null;

  return (
    <GlassSheet
      open
      onOpenChange={(next) => {
        if (!next) onClose();
      }}
      side="right"
      title="New run"
      description="Pick a capture, pick a recipe, override anything, and launch."
      className="admin-sheet"
      testId="new-run-sheet"
    >
      <div className="admin-form">
        <GlassField label="Capture" htmlFor={captureField}>
          <GlassSelect
            id={captureField}
            value={capture}
            onChange={(event) => {
              setCapture(event.target.value);
            }}
            data-testid="run-capture"
          >
            <option value="">Choose a capture…</option>
            {(captures.data ?? []).map((entry) => (
              <option key={entry.id} value={entry.id}>
                {entry.name} ({entry.slug})
              </option>
            ))}
          </GlassSelect>
        </GlassField>

        <GlassField
          label="Recipe"
          htmlFor={recipeField}
          hint={recipe ? `v${recipe.version} — ${recipe.description}` : undefined}
          error={
            catalogue.isError
              ? "No recipe catalogue on this API. Runs can still be queued by name from the globe app; this form needs the recipe files to offer their parameters."
              : undefined
          }
        >
          <GlassSelect
            id={recipeField}
            value={chosen}
            onChange={(event) => {
              setRecipeName(event.target.value);
              setDraft({});
            }}
            data-testid="run-recipe"
          >
            {recipes.map((entry) => (
              <option key={entry.name} value={entry.name}>
                {entry.name}
              </option>
            ))}
          </GlassSelect>
        </GlassField>

        <div className="admin-form__row">
          <GlassField label="Provider" htmlFor={providerField}>
            <GlassSelect
              id={providerField}
              value={provider}
              onChange={(event) => {
                setProvider(event.target.value);
                setTier("");
              }}
              data-testid="run-provider"
            >
              <option value="">Unset — the worker decides</option>
              {providers.map((entry) => (
                <option key={entry.name} value={entry.name}>
                  {entry.label}
                  {entry.interruptible ? " (interruptible)" : ""}
                </option>
              ))}
            </GlassSelect>
          </GlassField>

          <GlassField label="Tier" htmlFor={tierField}>
            <GlassSelect
              id={tierField}
              value={tier}
              onChange={(event) => {
                setTier(event.target.value);
              }}
              data-testid="run-tier"
            >
              <option value="">Unset</option>
              {(chosenProvider?.tiers ?? []).map((entry) => (
                <option key={entry} value={entry}>
                  {entry}
                </option>
              ))}
            </GlassSelect>
          </GlassField>
        </div>

        {chosenProvider && <p className="admin-note">{chosenProvider.note}</p>}

        <div className="admin-estimate" data-testid="estimate">
          <span className="admin-estimate__wall">{estimate.wall}</span>
          <span className="admin-estimate__cost">{estimate.cost}</span>
          <span className="admin-note">{estimate.basis}</span>
          {selected && (
            <span className="admin-note">
              {formatBytes(bytes)} of source across {selected.files.length} file
              {selected.files.length === 1 ? "" : "s"}
            </span>
          )}
        </div>

        <h3 className="admin-subhead">
          Parameters{" "}
          {changed > 0 && <span className="admin-dim">({changed} stage overridden)</span>}
        </h3>
        <p className="admin-note">
          Blank means the recipe&apos;s own default, shown as the placeholder. Only what you change
          is stored on the run.
        </p>

        {(recipe?.stages ?? []).map((stage) => (
          <fieldset key={stage.id} className="admin-stage">
            <legend>
              {stage.id} <span className="admin-mono admin-dim">{stage.impl}</span>
              {stage.gpu && <span className="admin-dim"> · GPU {stage.gpu.tier}</span>}
            </legend>
            {Object.keys(stage.params as Record<string, unknown>).length === 0 ? (
              <p className="admin-note">No parameters.</p>
            ) : (
              <div className="admin-stage__grid">
                {Object.entries(stage.params as Record<string, unknown>).map(([name, value]) => (
                  <label key={name} className="admin-param">
                    <span className="admin-param__name">{name}</span>
                    <GlassInput
                      mono
                      value={draft[stage.id]?.[name] ?? ""}
                      placeholder={JSON.stringify(value)}
                      aria-label={`${stage.id} ${name}`}
                      data-testid={`param-${stage.id}-${name}`}
                      onChange={(event) => {
                        const text = event.target.value;
                        setDraft((current) => ({
                          ...current,
                          [stage.id]: { ...current[stage.id], [name]: text },
                        }));
                      }}
                    />
                  </label>
                ))}
              </div>
            )}
          </fieldset>
        ))}

        {problem && (
          <p className="admin-error" role="alert" data-testid="launch-error">
            {problem}
          </p>
        )}

        <GlassButton
          variant="primary"
          leadingIcon={<Play size={13} aria-hidden="true" />}
          disabled={!capture || !chosen}
          loading={launch.isPending}
          data-testid="launch"
          onClick={() => {
            launch.mutate(
              {
                captureId: capture,
                body: {
                  recipe: chosen,
                  params,
                  provider: provider || null,
                  tier: tier || null,
                },
              },
              { onSuccess: onLaunched },
            );
          }}
        >
          Launch run
        </GlassButton>
      </div>
    </GlassSheet>
  );
}
