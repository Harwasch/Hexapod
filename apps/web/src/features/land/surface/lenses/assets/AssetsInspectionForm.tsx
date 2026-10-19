import { useEffect, useId, useRef, useState } from "react";
import { Plus, X } from "lucide-react";

import { GlassButton, GlassInput, GlassTextarea } from "@twin/ui";

import { api, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";

import { LandEvidencePicker } from "../../../LandEvidencePicker";
import { CONDITION_ORDER, CONDITIONS, type Condition } from "./assetsModel";

/** Now, as a `datetime-local` value in the person's own time zone. */
function localNow(): string {
  const now = new Date();
  return new Date(now.getTime() - now.getTimezoneOffset() * 60_000).toISOString().slice(0, 16);
}

interface Measurement {
  id: string;
  name: string;
  value: string;
  unit: string;
}

/**
 * Recording what someone saw: the condition, a line of notes and when. Measurements and linked
 * sources are one tap further. A retried save reuses its request key, so it is stored once.
 */
export function AssetsInspectionForm({
  landId,
  featureId,
  onSaved,
  onCancel,
}: {
  landId: string;
  featureId: string;
  onSaved: () => unknown;
  onCancel: () => void;
}) {
  const id = useId();
  const [condition, setCondition] = useState<Condition | null>(null);
  const [notes, setNotes] = useState("");
  const [observed, setObserved] = useState(localNow);
  const [measurements, setMeasurements] = useState<Measurement[]>([]);
  const [evidenceIds, setEvidenceIds] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestKey = useRef(crypto.randomUUID());
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const save = async () => {
    if (!condition) return;
    setBusy(true);
    setError(null);
    try {
      const time = new Date(observed);
      if (!Number.isFinite(time.getTime())) throw new Error("Choose a valid date and time.");
      const names = measurements.map((row) => row.name.trim());
      if (new Set(names).size !== names.length)
        throw new Error("Give each measurement a different name.");
      if (
        measurements.some(
          (row) =>
            !row.name.trim() ||
            !row.unit.trim() ||
            !row.value.trim() ||
            !Number.isFinite(Number(row.value)),
        )
      )
        throw new Error("Each measurement needs a name, a number and a unit.");
      await unwrap(
        api.POST("/api/v1/land/{land_id}/features/{feature_id}/inspections", {
          params: { path: { land_id: landId, feature_id: featureId } },
          body: {
            requestKey: requestKey.current,
            observedAt: time.toISOString(),
            condition,
            notes: notes.trim(),
            measurements: Object.fromEntries(
              measurements.map((row) => [row.name.trim(), Number(row.value)]),
            ),
            measurementUnits: Object.fromEntries(
              measurements.map((row) => [row.name.trim(), row.unit.trim()]),
            ),
            ...(evidenceIds.length ? { evidenceIds } : {}),
          },
        }),
      );
      if (mounted.current) requestKey.current = crypto.randomUUID();
      await onSaved();
    } catch (cause) {
      if (mounted.current) setError(describeError(cause));
    } finally {
      if (mounted.current) setBusy(false);
    }
  };

  return (
    <form
      className="assets-form"
      aria-label="Record an inspection"
      onSubmit={(event) => {
        event.preventDefault();
        void save();
      }}
    >
      {error && (
        <p className="land-alert" role="alert">
          {error}
        </p>
      )}
      <fieldset disabled={busy} className="assets-form__fields">
        <fieldset className="assets-chips">
          <legend className="land-field">Condition</legend>
          {CONDITION_ORDER.map((value) => (
            <label key={value} className="assets-chip">
              <input
                type="radio"
                name={`${id}-condition`}
                value={value}
                checked={condition === value}
                onChange={() => setCondition(value)}
              />
              <i style={{ background: CONDITIONS[value].color }} aria-hidden="true" />
              {CONDITIONS[value].label}
            </label>
          ))}
        </fieldset>
        <label className="land-field" htmlFor={`${id}-notes`}>
          What did you see?
          <GlassTextarea
            id={`${id}-notes`}
            required
            rows={2}
            maxLength={10000}
            value={notes}
            placeholder="Posts leaning near the creek crossing"
            onChange={(event) => setNotes(event.target.value)}
          />
        </label>
        <label className="land-field" htmlFor={`${id}-when`}>
          When
          <GlassInput
            id={`${id}-when`}
            type="datetime-local"
            required
            value={observed}
            onChange={(event) => setObserved(event.target.value)}
          />
        </label>
        <details className="land-details">
          <summary>
            Measurements and sources
            {measurements.length + evidenceIds.length
              ? ` · ${measurements.length + evidenceIds.length}`
              : ""}
          </summary>
          <div className="land-details__body">
            {measurements.map((row, index) => (
              <div key={row.id} className="assets-measure">
                {(["name", "value", "unit"] as const).map((field) => (
                  <GlassInput
                    key={field}
                    required
                    maxLength={field === "value" ? 30 : 100}
                    inputMode={field === "value" ? "decimal" : "text"}
                    placeholder={field === "name" ? "Height" : field === "value" ? "2.4" : "m"}
                    aria-label={`Measurement ${index + 1} ${field}`}
                    value={row[field]}
                    onChange={(event) =>
                      setMeasurements(
                        measurements.map((item) =>
                          item.id === row.id ? { ...item, [field]: event.target.value } : item,
                        ),
                      )
                    }
                  />
                ))}
                <GlassButton
                  iconOnly
                  size="sm"
                  variant="ghost"
                  aria-label={`Remove measurement ${index + 1}`}
                  onClick={() => setMeasurements(measurements.filter((item) => item.id !== row.id))}
                >
                  <X size={14} aria-hidden="true" />
                </GlassButton>
              </div>
            ))}
            <div className="land-row">
              <GlassButton
                size="sm"
                disabled={measurements.length >= 100}
                leadingIcon={<Plus size={14} aria-hidden="true" />}
                onClick={() =>
                  setMeasurements([
                    ...measurements,
                    { id: crypto.randomUUID(), name: "", value: "", unit: "" },
                  ])
                }
              >
                Add measurement
              </GlassButton>
            </div>
            <div className="assets-tool">
              <LandEvidencePicker
                landId={landId}
                value={evidenceIds}
                onChange={setEvidenceIds}
                label="Sources"
                limit={100}
              />
            </div>
          </div>
        </details>
        <div className="land-row">
          <GlassButton
            type="submit"
            variant="primary"
            loading={busy}
            disabled={!condition || !notes.trim() || busy}
          >
            Save inspection
          </GlassButton>
          <GlassButton variant="ghost" onClick={onCancel}>
            Cancel
          </GlassButton>
        </div>
      </fieldset>
    </form>
  );
}
