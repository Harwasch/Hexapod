import { clsx } from "clsx";

export interface SpinnerProps {
  className?: string;
  label?: string;
}

export function Spinner({ className, label = "Loading" }: SpinnerProps) {
  return <span role="status" aria-label={label} className={clsx("glass-spinner", className)} />;
}
