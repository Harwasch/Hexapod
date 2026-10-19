import { cloneElement, useId, type ReactElement, type ReactNode } from "react";

/** A label and the one control it names (the control receives the id the label points at). */
export function PlanField({
  label,
  children,
}: {
  label: ReactNode;
  children: ReactElement<{ id?: string }>;
}) {
  const id = useId();
  return (
    <label className="land-field" htmlFor={id}>
      {label}
      {cloneElement(children, { id })}
    </label>
  );
}
