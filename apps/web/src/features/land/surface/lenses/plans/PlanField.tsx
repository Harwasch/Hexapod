import { cloneElement, useId, type ReactElement, type ReactNode } from "react";

/**
 * A label and the one control it names. A control with an id of its own (a field the save
 * checklist points at) keeps it; any other receives one for the label to point at.
 */
export function PlanField({
  label,
  children,
}: {
  label: ReactNode;
  children: ReactElement<{ id?: string }>;
}) {
  const generated = useId();
  const id = children.props.id ?? generated;
  return (
    <label className="land-field" htmlFor={id}>
      {label}
      {cloneElement(children, { id })}
    </label>
  );
}
