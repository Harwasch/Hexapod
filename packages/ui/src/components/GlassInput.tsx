import { clsx } from "clsx";
import {
  forwardRef,
  type InputHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
  type TextareaHTMLAttributes,
} from "react";

export interface GlassFieldProps {
  label: ReactNode;
  hint?: ReactNode;
  error?: ReactNode;
  htmlFor?: string;
  children: ReactNode;
  className?: string;
}

/** Label + control + hint/error wrapper. Pass the control's id via `htmlFor`. */
export function GlassField({ label, hint, error, htmlFor, children, className }: GlassFieldProps) {
  return (
    <div className={clsx("glass-field", className)}>
      <label className="glass-field__label" htmlFor={htmlFor}>
        {label}
      </label>
      {children}
      {error ? (
        <p className="glass-field__error" role="alert">
          {error}
        </p>
      ) : hint ? (
        <p className="glass-field__hint">{hint}</p>
      ) : null}
    </div>
  );
}

export interface GlassInputProps extends InputHTMLAttributes<HTMLInputElement> {
  invalid?: boolean;
  mono?: boolean;
}

export const GlassInput = forwardRef<HTMLInputElement, GlassInputProps>(function GlassInput(
  { invalid = false, mono = false, className, ...rest },
  ref,
) {
  return (
    <input
      ref={ref}
      aria-invalid={invalid || undefined}
      className={clsx("glass-input", mono && "glass-input--mono", className)}
      {...rest}
    />
  );
});

export interface GlassTextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  invalid?: boolean;
  mono?: boolean;
}

export const GlassTextarea = forwardRef<HTMLTextAreaElement, GlassTextareaProps>(
  function GlassTextarea({ invalid = false, mono = false, className, ...rest }, ref) {
    return (
      <textarea
        ref={ref}
        aria-invalid={invalid || undefined}
        className={clsx("glass-input", mono && "glass-input--mono", className)}
        {...rest}
      />
    );
  },
);

export type GlassSelectProps = SelectHTMLAttributes<HTMLSelectElement>;

export const GlassSelect = forwardRef<HTMLSelectElement, GlassSelectProps>(function GlassSelect(
  { className, ...rest },
  ref,
) {
  return <select ref={ref} className={clsx("glass-input", className)} {...rest} />;
});
