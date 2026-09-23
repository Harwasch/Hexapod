import * as Switch from "@radix-ui/react-switch";
import { clsx } from "clsx";

export interface GlassSwitchProps {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  disabled?: boolean;
  id?: string;
  "aria-label"?: string;
  "aria-labelledby"?: string;
  className?: string;
}

export function GlassSwitch({
  checked,
  onCheckedChange,
  disabled,
  id,
  className,
  ...aria
}: GlassSwitchProps) {
  return (
    <Switch.Root
      id={id}
      checked={checked}
      onCheckedChange={onCheckedChange}
      disabled={disabled}
      className={clsx("glass-switch", className)}
      {...aria}
    >
      <Switch.Thumb className="glass-switch__thumb" />
    </Switch.Root>
  );
}
