import { useState } from "react";
import { Plus, X } from "lucide-react";

import { GlassButton, GlassInput } from "@twin/ui";

type Value = string | number | boolean | null;

interface Row {
  id: string;
  key: string;
  value: Value;
}

function text(value: Value): string {
  return value === null ? "" : String(value);
}

/**
 * Every recorded attribute as a name and a value. Values keep their type until edited; an
 * edited value is saved as text. Rows without a name are not saved.
 */
export function AssetsAttributes({
  value,
  onChange,
}: {
  value: Record<string, Value> | undefined;
  onChange: (value: Record<string, Value>) => void;
}) {
  const [rows, setRows] = useState<Row[]>(() =>
    Object.entries(value ?? {}).map(([key, item]) => ({
      id: crypto.randomUUID(),
      key,
      value: item,
    })),
  );
  const names = rows.map((row) => row.key.trim()).filter(Boolean);
  const duplicate = new Set(names).size !== names.length;
  const push = (next: Row[]) => {
    setRows(next);
    onChange(
      Object.fromEntries(
        next.filter((row) => row.key.trim()).map((row) => [row.key.trim(), row.value]),
      ),
    );
  };
  return (
    <fieldset className="assets-attributes">
      <legend className="land-field">Attributes</legend>
      {rows.map((row, index) => (
        <div key={row.id} className="assets-attributes__row">
          <GlassInput
            aria-label={`Attribute ${index + 1} name`}
            placeholder="Name"
            maxLength={100}
            value={row.key}
            onChange={(event) =>
              push(
                rows.map((item) =>
                  item.id === row.id ? { ...item, key: event.target.value } : item,
                ),
              )
            }
          />
          <GlassInput
            aria-label={`Attribute ${index + 1} value`}
            placeholder="Value"
            maxLength={2000}
            value={text(row.value)}
            onChange={(event) =>
              push(
                rows.map((item) =>
                  item.id === row.id ? { ...item, value: event.target.value } : item,
                ),
              )
            }
          />
          <GlassButton
            iconOnly
            size="sm"
            variant="ghost"
            aria-label={`Remove attribute ${index + 1}`}
            onClick={() => push(rows.filter((item) => item.id !== row.id))}
          >
            <X size={14} aria-hidden="true" />
          </GlassButton>
        </div>
      ))}
      {duplicate && (
        <p className="land-alert" role="alert">
          Two attributes share a name; only the last one is kept.
        </p>
      )}
      <div className="land-row">
        <GlassButton
          size="sm"
          disabled={rows.length >= 100}
          leadingIcon={<Plus size={14} aria-hidden="true" />}
          onClick={() => setRows([...rows, { id: crypto.randomUUID(), key: "", value: "" }])}
        >
          Add attribute
        </GlassButton>
      </div>
    </fieldset>
  );
}
