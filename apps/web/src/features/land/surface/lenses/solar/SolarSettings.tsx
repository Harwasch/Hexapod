import { useId } from "react";

import { GlassButton, GlassInput, GlassSelect, GlassTextarea } from "@twin/ui";

import {
  MOUNTINGS,
  settingsSummary,
  shownValue,
  solarFields,
  storedValue,
  type SolarMounting,
  type SolarRequest,
} from "./solarDraft";

/**
 * Every assumption of the hourly model, folded under one line that says what they are now.
 * Each starts at a planning default, so most people never open it.
 */
export function SolarSettings({
  request,
  onChange,
  disabled,
}: {
  request: SolarRequest;
  onChange: (next: SolarRequest) => void;
  disabled?: boolean;
}) {
  const id = useId();
  const horizon = request.horizon ?? [];
  const text = (key: "zoneBasis" | "horizonBasis" | "assumptions", label: string, max: number) => (
    <label className="land-field" key={key}>
      {label}
      <GlassTextarea
        rows={2}
        maxLength={max}
        required
        value={request[key]}
        onChange={(event) => onChange({ ...request, [key]: event.target.value })}
      />
    </label>
  );
  return (
    <details className="land-details land-solar-settings">
      <summary>
        <span className="land-solar-settings__summary">
          <span>System settings</span>
          <span className="land-muted">{settingsSummary(request)}</span>
        </span>
      </summary>
      <fieldset className="land-solar-settings__body" disabled={disabled}>
        <legend className="land-visually-hidden">System settings</legend>
        <div className="land-solar-grid">
          {solarFields().map((field) => (
            <div className="land-field" key={field.key}>
              <label htmlFor={`${id}-${field.key}`}>{field.label}</label>
              <GlassInput
                id={`${id}-${field.key}`}
                type="number"
                step={field.key === "year" ? 1 : "any"}
                inputMode="decimal"
                aria-describedby={field.hint ? `${id}-${field.key}-hint` : undefined}
                value={shownValue(field, request[field.key])}
                onChange={(event) =>
                  onChange({ ...request, [field.key]: storedValue(field, event.target.value) })
                }
              />
              {field.hint && (
                <span className="land-solar-hint" id={`${id}-${field.key}-hint`}>
                  {field.hint}
                </span>
              )}
            </div>
          ))}
        </div>
        <label className="land-field">
          Mounting
          <GlassSelect
            value={request.mounting}
            onChange={(event) =>
              onChange({ ...request, mounting: event.target.value as SolarMounting })
            }
          >
            {MOUNTINGS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </GlassSelect>
        </label>
        <div className="land-solar-horizon" role="group" aria-label="Horizon">
          <span className="land-field">Horizon</span>
          <div className="land-row">
            <GlassButton
              size="sm"
              variant="ghost"
              active={!horizon.length}
              onClick={() => onChange({ ...request, horizon: [] })}
            >
              Open horizon
            </GlassButton>
            <GlassButton
              size="sm"
              variant="ghost"
              active={horizon.length > 0}
              onClick={() =>
                onChange({
                  ...request,
                  horizon: horizon.length
                    ? horizon
                    : [0, 90, 180, 270].map((azimuthDegrees) => ({
                        azimuthDegrees,
                        elevationDegrees: 0,
                      })),
                })
              }
            >
              Enter four directions
            </GlassButton>
          </div>
          <span className="land-solar-hint">
            Height is the angle of hills or trees above flat, not their height in metres.
          </span>
          {horizon.map((point, index) => (
            <div className="land-solar-grid" key={index}>
              <label className="land-field">
                Direction {index + 1} (degrees from north)
                <GlassInput
                  type="number"
                  step="any"
                  min={0}
                  max={359.999}
                  value={Number.isFinite(point.azimuthDegrees) ? point.azimuthDegrees : ""}
                  onChange={(event) =>
                    onChange({
                      ...request,
                      horizon: horizon.map((item, at) =>
                        at === index
                          ? { ...item, azimuthDegrees: Number(event.target.value) }
                          : item,
                      ),
                    })
                  }
                />
              </label>
              <label className="land-field">
                Height {index + 1} (degrees)
                <GlassInput
                  type="number"
                  step="any"
                  min={0}
                  max={90}
                  value={Number.isFinite(point.elevationDegrees) ? point.elevationDegrees : ""}
                  onChange={(event) =>
                    onChange({
                      ...request,
                      horizon: horizon.map((item, at) =>
                        at === index
                          ? { ...item, elevationDegrees: Number(event.target.value) }
                          : item,
                      ),
                    })
                  }
                />
              </label>
            </div>
          ))}
        </div>
        {text("zoneBasis", "Where the outline came from", 2000)}
        {text("horizonBasis", "Where the horizon came from", 2000)}
        {text("assumptions", "Equipment and other assumptions", 5000)}
      </fieldset>
    </details>
  );
}

/** The settings of a saved assessment, read only. */
export function SolarSettingsList({ request }: { request: SolarRequest }) {
  const fields = solarFields();
  return (
    <dl className="land-solar-list">
      {fields.map((field) => (
        <div key={field.key}>
          <dt>{field.label.replace(/ \(.*\)$/, "")}</dt>
          <dd>
            {shownValue(field, request[field.key])}
            {field.unit}
          </dd>
        </div>
      ))}
      <div>
        <dt>Mounting</dt>
        <dd>{MOUNTINGS.find((option) => option.value === request.mounting)?.label}</dd>
      </div>
      <div>
        <dt>Horizon</dt>
        <dd>
          {request.horizon?.length
            ? request.horizon
                .map((point) => `${point.azimuthDegrees}° at ${point.elevationDegrees}° high`)
                .join(", ")
            : "Open"}
        </dd>
      </div>
    </dl>
  );
}
