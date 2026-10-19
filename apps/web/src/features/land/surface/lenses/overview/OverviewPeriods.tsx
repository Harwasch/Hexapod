import { useState } from "react";
import { Plus, X } from "lucide-react";

import { GlassButton, GlassInput } from "@twin/ui";

import { vegetationPeriods } from "../../../vegetationPeriods";
import { defaultMonths, type Period } from "./overviewData";
import "./overview.css";

/**
 * Chooses the months a greenness comparison looks at: up to six, last month and a year before
 * to start with. Each month picks one clear satellite pass after checking clouds.
 */
export function OverviewPeriods({
  disabled,
  busy,
  onStart,
  startLabel = "Compare these months",
  initial,
}: {
  disabled: boolean;
  busy: boolean;
  onStart: (periods: Period[]) => void;
  startLabel?: string;
  /** Months to start from (YYYY-MM); last month and the same month a year before otherwise. */
  initial?: string[];
}) {
  const [months, setMonths] = useState(() => (initial?.length ? initial : defaultMonths()));
  const [latest] = useState(() => new Date().toISOString().slice(0, 7));
  const periods = vegetationPeriods(months);
  return (
    <div className="overview-periods">
      <p className="land-muted">
        Up to six months. The same month in different years compares best; cloudy months stay as
        gaps.
      </p>
      <ol className="overview-periods__list">
        {months.map((month, index) => (
          <li key={index}>
            <label className="land-field">
              Observation month {index + 1}
              <GlassInput
                type="month"
                min="2015-07"
                max={latest}
                value={month}
                disabled={busy}
                onChange={(event) =>
                  setMonths(months.map((value, i) => (i === index ? event.target.value : value)))
                }
              />
            </label>
            <GlassButton
              size="sm"
              variant="ghost"
              iconOnly
              aria-label={`Remove observation month ${index + 1}`}
              disabled={busy || months.length === 1}
              onClick={() => setMonths(months.filter((_, i) => i !== index))}
            >
              <X size={14} aria-hidden="true" />
            </GlassButton>
          </li>
        ))}
      </ol>
      {!periods && (
        <p className="land-muted" role="status">
          Choose different months from July 2015 to this month.
        </p>
      )}
      <div className="land-row">
        <GlassButton
          size="sm"
          variant="ghost"
          leadingIcon={<Plus size={14} aria-hidden="true" />}
          disabled={busy || months.length >= 6}
          onClick={() => setMonths([...months, ""])}
        >
          Add a month
        </GlassButton>
        <GlassButton
          size="sm"
          disabled={disabled || !periods}
          loading={busy}
          onClick={() => {
            if (periods) onStart(periods);
          }}
        >
          {startLabel}
        </GlassButton>
      </div>
    </div>
  );
}
