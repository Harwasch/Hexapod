import { Crosshair, Ruler } from "lucide-react";
import { useEffect, useRef, useState, type SubmitEvent } from "react";

import type { AssetScaleUpdate, Site, SiteAsset } from "@twin/contracts";
import { formatLength } from "@twin/geo";
import { GlassButton, GlassField, GlassInput, useFieldId } from "@twin/ui";

import { isUnauthorized } from "@/api/client";
import { useSetAssetScale } from "@/api/queries";
import type { ScaleMeasureState } from "@/cesium/ScaleMeasure";
import { useScene } from "@/cesium/SceneContext";
import {
  assetScale,
  directScaleBody,
  formatScale,
  isScale,
  measuredScaleBody,
  parseLength,
  parseScale,
  RESET_SCALE_BODY,
  scalableAsset,
  scaleFromLength,
} from "@/lib/realSize";
import { useSettings } from "@/state/settings";

import { WriteTokenField } from "../captures/WriteTokenField";

/** What Save would send: a measured length, a factor typed in, or back to the registered model. */
type Pending =
  | {
      readonly kind: "measured";
      readonly scale: number;
      readonly measuredM: number;
      readonly trueM: number;
      readonly atScale: number;
    }
  | { readonly kind: "direct"; readonly scale: number }
  | { readonly kind: "reset"; readonly scale: number };

/** A length measured on the scan, as drawn when its second point was placed. */
interface Measured {
  readonly lengthM: number;
  readonly atScale: number;
}

const TOKEN_HINT =
  "This server needs a token to save a scan's size. It is stored in this browser, which suits a single-user setup.";

function bodyFor(pending: Pending): AssetScaleUpdate {
  if (pending.kind === "measured") {
    return measuredScaleBody({
      measuredM: pending.measuredM,
      trueM: pending.trueM,
      atScale: pending.atScale,
    });
  }
  return pending.kind === "direct" ? directScaleBody(pending.scale) : RESET_SCALE_BODY;
}

/**
 * Set real size: the site card's tool for a scan drawn at the wrong size -- a phone video
 * reconstructed with no metric scale is drawn at one of its units to the metre, two to eight
 * times too big (docs/DATA_MODEL.md "Runtime scale").
 *
 * Measure on the scan: two points picked on it (cesium/ScaleMeasure.ts) give a length as drawn
 * now; the true length typed in makes the scale `drawn-at × true ÷ measured`. Or the factor is
 * typed in directly, or the scan is reset to the model as registered. Every change is previewed
 * live (`SiteManager.previewScale`), nothing is saved until Save (`PUT /assets/{id}/scale`), and
 * Escape or Cancel puts the scan back as it was. A 401 asks for the write token and tries again.
 *
 * Offered only for the scan the API can resize (`scalableAsset`): the splat a pipeline run
 * registered, whose site records the origin its tiles are placed at.
 */
export function RealSize({ site }: { site: Site }) {
  const asset = scalableAsset(site);
  if (!asset) return null;
  return <RealSizeTool key={asset.id} site={site} asset={asset} />;
}

function RealSizeTool({ site, asset }: { site: Site; asset: SiteAsset }) {
  const scene = useScene();
  const units = useSettings((s) => s.units);
  const saved = assetScale(asset);
  const mutation = useSetAssetScale();
  const trueId = useFieldId("real-size-true");
  const scaleId = useFieldId("real-size-scale");

  const [open, setOpen] = useState(false);
  const [measure, setMeasure] = useState<ScaleMeasureState | null>(null);
  const [measured, setMeasured] = useState<Measured | null>(null);
  const [trueText, setTrueText] = useState("");
  const [scaleText, setScaleText] = useState("");
  const [pending, setPending] = useState<Pending | null>(null);
  const [needsToken, setNeedsToken] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  // The true length as typed, for a measurement that lands after it was typed.
  const typed = useRef("");
  // Focus goes into the tool as it opens and back to its button as it closes.
  const openButton = useRef<HTMLButtonElement>(null);
  const measureButton = useRef<HTMLButtonElement>(null);
  const wasOpen = useRef(false);
  useEffect(() => {
    if (open) measureButton.current?.focus();
    else if (wasOpen.current) openButton.current?.focus();
    wasOpen.current = open;
  }, [open]);
  // What the unmount clean-up needs, current: the scene, and whether a preview is showing.
  const live = useRef({ scene, open, siteId: site.id });
  useEffect(() => {
    live.current = { scene, open, siteId: site.id };
  });

  // The preview: the scan drawn at what Save would send, or as saved.
  const previewed = pending?.scale ?? null;
  useEffect(() => {
    if (!open) return;
    scene?.sites.previewScale(site.id, previewed);
  }, [scene, open, site.id, previewed]);

  // Leaving the card mid-edit (the inspector closed, another site picked) is a cancel.
  useEffect(
    () => () => {
      const { scene: current, open: editing, siteId } = live.current;
      if (!editing) return;
      current?.scaleMeasure.clear();
      current?.sites.previewScale(siteId, null);
    },
    [],
  );

  const reset = (): void => {
    scene?.scaleMeasure.clear();
    setMeasure(null);
    setMeasured(null);
    typed.current = "";
    setTrueText("");
    setScaleText("");
    setPending(null);
    setNeedsToken(false);
    setNotice(null);
    mutation.reset();
    setOpen(false);
  };

  /** Escape, Cancel: the scan back at its saved scale, nothing sent. */
  const cancel = (): void => {
    scene?.sites.previewScale(site.id, null);
    reset();
  };
  const cancelRef = useRef(cancel);
  useEffect(() => {
    cancelRef.current = cancel;
  });

  // Escape cancels the edit, ahead of the app's own step back (which would close the card and
  // leave the preview showing): a capturing listener, which marks the key as handled.
  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent): void => {
      if (event.key !== "Escape" || event.defaultPrevented) return;
      event.preventDefault();
      cancelRef.current();
    };
    window.addEventListener("keydown", onKey, { capture: true });
    return () => window.removeEventListener("keydown", onKey, { capture: true });
  }, [open]);

  // A bare number is in the unit lengths are read in here: metres, or feet.
  const bare = units === "imperial" ? "ft" : "m";

  /** The scale a measured length and a true length typed say, if both are there and valid. */
  const fromLength = (length: Measured | null, text: string): Pending | null => {
    const trueM = parseLength(text, bare);
    if (!length || trueM === null) return null;
    const scale = scaleFromLength(length.lengthM, trueM, length.atScale);
    if (!isScale(scale)) return null;
    return { kind: "measured", scale, measuredM: length.lengthM, trueM, atScale: length.atScale };
  };

  const startMeasuring = (): void => {
    setNotice(null);
    setMeasured(null);
    const started = scene?.scaleMeasure.start(site.id, (state) => {
      setMeasure(state);
      if (state.points < 2 || state.lengthM === null) {
        setMeasured(null);
        return;
      }
      // Measured as the scan is drawn now -- a preview included -- and said so in the evidence.
      const atScale = scene.sites.shownScale(site.id) ?? saved;
      const length = { lengthM: state.lengthM, atScale };
      setMeasured(length);
      const next = fromLength(length, typed.current);
      if (next) setPending(next);
    });
    if (!started) setNotice("Fly to the scan first: it has to be in the scene to be measured on.");
  };

  const onTrue = (text: string): void => {
    typed.current = text;
    setTrueText(text);
    setScaleText("");
    setPending(fromLength(measured, text));
  };

  const onScale = (text: string): void => {
    // Whichever is edited last is what Save sends: a factor typed puts the true length aside.
    typed.current = "";
    setTrueText("");
    setScaleText(text);
    const scale = parseScale(text);
    setPending(scale === null ? null : { kind: "direct", scale });
  };

  const onReset = (): void => {
    setScaleText("");
    typed.current = "";
    setTrueText("");
    setPending({ kind: "reset", scale: 1 });
  };

  const save = async (): Promise<void> => {
    if (!pending) return;
    setNotice(null);
    try {
      await mutation.mutateAsync({ siteId: site.id, assetId: asset.id, body: bodyFor(pending) });
      // The saved record draws the scan at this scale (`SiteManager.updateRecord`): the
      // preview is not undone first, which would show the old size for a moment.
      reset();
    } catch (error) {
      if (isUnauthorized(error)) setNeedsToken(true);
    }
  };

  const submit = (event: SubmitEvent<HTMLFormElement>): void => {
    event.preventDefault();
    void save();
  };

  if (!open) {
    return (
      <section aria-label="Real size" className="real-size" data-testid="real-size">
        <div className="glass-row glass-row--between">
          <p className="glass-eyebrow">Real size · {site.name}</p>
          <span className="glass-mono glass-subtle" data-testid="real-size-current">
            {saved === 1 ? "as registered" : formatScale(saved)}
          </span>
        </div>
        <GlassButton
          ref={openButton}
          size="sm"
          leadingIcon={<Ruler size={13} aria-hidden="true" />}
          onClick={() => setOpen(true)}
          data-testid="real-size-open"
        >
          Set real size
        </GlassButton>
      </section>
    );
  }

  const picking = measure?.picking ?? false;
  const unchanged = pending === null || Math.abs(pending.scale - saved) < 1e-9;
  const error =
    mutation.error && !isUnauthorized(mutation.error)
      ? `Not saved: ${mutation.error instanceof Error ? mutation.error.message : String(mutation.error)}`
      : null;

  return (
    <section aria-label="Real size" className="real-size" data-testid="real-size">
      <div className="glass-row glass-row--between">
        <p className="glass-eyebrow">Real size · {site.name}</p>
        <span className="glass-mono glass-subtle">now {formatScale(saved)}</span>
      </div>
      <form className="real-size__form" onSubmit={submit} noValidate>
        <div className="real-size__step">
          <GlassButton
            ref={measureButton}
            size="sm"
            type="button"
            active={picking}
            leadingIcon={<Ruler size={13} aria-hidden="true" />}
            onClick={startMeasuring}
            data-testid="real-size-measure"
          >
            {measured ? "Measure again" : "Measure on the scan"}
          </GlassButton>
          {picking && (
            <>
              <p className="real-size__note" role="status" data-testid="real-size-picking">
                Click or tap two points on the scan a known distance apart (point{" "}
                {(measure?.points ?? 0) + 1} of 2).
                {measure?.missed && " That was not on the scan: try on the scan itself."}
              </p>
              <GlassButton
                size="sm"
                type="button"
                variant="ghost"
                leadingIcon={<Crosshair size={13} aria-hidden="true" />}
                onClick={() => scene?.scaleMeasure.markCentre()}
                data-testid="real-size-centre"
              >
                Mark the centre of the view
              </GlassButton>
            </>
          )}
          {measured && (
            <p className="real-size__note" data-testid="real-size-measured">
              Measured <strong>{formatLength(measured.lengthM, "metric")}</strong>
              {units === "imperial" && ` (${formatLength(measured.lengthM, units)})`} on the scan as
              drawn at {formatScale(measured.atScale)}.
            </p>
          )}
          <GlassField
            label="True length"
            htmlFor={trueId}
            hint={
              bare === "ft"
                ? "Feet unless it says otherwise: 72 in, 1.8 m."
                : "Metres unless it says otherwise: 180 cm, 6 ft."
            }
          >
            <GlassInput
              id={trueId}
              mono
              inputMode="decimal"
              autoComplete="off"
              placeholder={
                measured ? (bare === "ft" ? "e.g. 6 ft" : "e.g. 1.8 m") : "Measure first"
              }
              disabled={!measured}
              value={trueText}
              invalid={trueText !== "" && pending?.kind !== "measured"}
              onChange={(event) => onTrue(event.target.value)}
              data-testid="real-size-true"
            />
          </GlassField>
        </div>
        <GlassField
          label="Or the scale, × as registered"
          htmlFor={scaleId}
          hint="0.01 to 100, absolute: 0.5 is half the size it was registered at, whatever it is now."
        >
          <GlassInput
            id={scaleId}
            mono
            inputMode="decimal"
            autoComplete="off"
            placeholder={formatScale(saved).slice(1)}
            value={scaleText}
            invalid={scaleText !== "" && pending?.kind !== "direct"}
            onChange={(event) => onScale(event.target.value)}
            data-testid="real-size-scale"
          />
        </GlassField>
        {pending && (
          <p className="real-size__note" role="status" data-testid="real-size-preview">
            {pending.kind === "reset" ? "Previewing the scan as registered" : "Previewing"}{" "}
            <strong className="glass-mono">{formatScale(pending.scale)}</strong>. Save to keep it,
            Esc to cancel.
          </p>
        )}
        {notice && (
          <p className="real-size__note" role="status">
            {notice}
          </p>
        )}
        {error && (
          <p className="glass-field__error" role="alert" data-testid="real-size-error">
            {error}
          </p>
        )}
        <div className="glass-row real-size__actions">
          <GlassButton
            size="sm"
            type="submit"
            variant="primary"
            disabled={unchanged}
            loading={mutation.isPending}
            data-testid="real-size-save"
          >
            Save
          </GlassButton>
          <GlassButton size="sm" type="button" onClick={cancel} data-testid="real-size-cancel">
            Cancel
          </GlassButton>
          <GlassButton
            size="sm"
            type="button"
            variant="ghost"
            disabled={saved === 1 && pending === null}
            onClick={onReset}
            data-testid="real-size-reset"
          >
            Reset to registered
          </GlassButton>
        </div>
      </form>
      {needsToken && <WriteTokenField hint={TOKEN_HINT} onSaved={() => void save()} />}
    </section>
  );
}
