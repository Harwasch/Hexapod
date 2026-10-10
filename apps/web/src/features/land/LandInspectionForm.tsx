import { useEffect, useRef, useState } from "react";
import type { components } from "@twin/contracts";
import { api, unwrap } from "@/api/client";
import { describeError } from "@/lib/log";

type Inspection = components["schemas"]["FeatureInspectionCreate"];

export function LandInspectionForm({
  landId,
  featureId,
  onSaved,
}: {
  landId: string;
  featureId: string;
  onSaved: () => Promise<unknown>;
}) {
  const [condition, setCondition] = useState<Inspection["condition"]>("unknown");
  const [notes, setNotes] = useState("");
  const [observed, setObserved] = useState(() => new Date().toISOString().slice(0, 16));
  const [measurements, setMeasurements] = useState<
    { id: string; name: string; value: string; unit: string }[]
  >([]);
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
    setBusy(true);
    setError(null);
    try {
      const time = new Date(`${observed}Z`);
      if (!Number.isFinite(time.getTime()))
        throw new Error("Choose a valid observation date and time.");
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
        throw new Error("Each measurement needs a name, finite value and unit.");
      await unwrap(
        api.POST("/api/v1/land/{land_id}/features/{feature_id}/inspections", {
          params: { path: { land_id: landId, feature_id: featureId } },
          body: {
            requestKey: requestKey.current,
            observedAt: time.toISOString(),
            condition,
            notes,
            measurements: Object.fromEntries(
              measurements.map((row) => [row.name.trim(), Number(row.value)]),
            ),
            measurementUnits: Object.fromEntries(
              measurements.map((row) => [row.name.trim(), row.unit.trim()]),
            ),
          },
        }),
      );
      if (mounted.current) {
        setNotes("");
        setMeasurements([]);
        requestKey.current = crypto.randomUUID();
      }
      await onSaved();
    } catch (cause) {
      if (mounted.current) setError(describeError(cause));
    } finally {
      if (mounted.current) setBusy(false);
    }
  };
  return (
    <form
      className="land-inventory-form"
      onSubmit={(event) => {
        event.preventDefault();
        void save();
      }}
    >
      {error && <p role="alert">{error}</p>}
      <fieldset disabled={busy}>
        <legend>New inspection</legend>
        <label className="land-name">
          Observed at (UTC)
          <input
            type="datetime-local"
            required
            value={observed}
            onChange={(event) => setObserved(event.target.value)}
          />
        </label>
        <label className="land-name">
          Condition
          <select
            aria-label="Condition"
            value={condition}
            onChange={(event) => setCondition(event.target.value as Inspection["condition"])}
          >
            {["unknown", "good", "fair", "poor", "critical"].map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
        </label>
        <label className="land-name">
          Inspection notes
          <textarea
            required
            maxLength={10000}
            rows={3}
            value={notes}
            onChange={(event) => setNotes(event.target.value)}
          />
        </label>
        {measurements.map((row, index) => (
          <div key={row.id} className="land-measurement-row">
            {(["name", "value", "unit"] as const).map((field) => (
              <label className="land-name" key={field}>
                {field === "name" ? "Measurement" : field === "value" ? "Value" : "Unit"}
                <input
                  required
                  maxLength={field === "value" ? 30 : 100}
                  inputMode={field === "value" ? "decimal" : "text"}
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
              </label>
            ))}
            <button
              type="button"
              aria-label={`Remove measurement ${index + 1}`}
              onClick={() => setMeasurements(measurements.filter((item) => item.id !== row.id))}
            >
              Remove
            </button>
          </div>
        ))}
        <div className="land-actions">
          <button
            type="button"
            disabled={measurements.length >= 100}
            onClick={() =>
              setMeasurements([
                ...measurements,
                { id: crypto.randomUUID(), name: "", value: "", unit: "" },
              ])
            }
          >
            Add measurement
          </button>
          <button disabled={!notes.trim()}>Record inspection</button>
        </div>
      </fieldset>
    </form>
  );
}
