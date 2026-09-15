import * as Slider from "@radix-ui/react-slider";
import { clsx } from "clsx";

export interface GlassSliderProps {
  value: number;
  onValueChange: (value: number) => void;
  onValueCommit?: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
  "aria-label": string;
  className?: string;
}

export function GlassSlider({
  value,
  onValueChange,
  onValueCommit,
  min = 0,
  max = 1,
  step = 0.01,
  disabled,
  className,
  ...aria
}: GlassSliderProps) {
  return (
    <Slider.Root
      className={clsx("glass-slider", className)}
      value={[value]}
      onValueChange={([next]) => {
        if (next !== undefined) onValueChange(next);
      }}
      onValueCommit={([next]) => {
        if (next !== undefined) onValueCommit?.(next);
      }}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
    >
      <Slider.Track className="glass-slider__track">
        <Slider.Range className="glass-slider__range" />
      </Slider.Track>
      <Slider.Thumb className="glass-slider__thumb" aria-label={aria["aria-label"]} />
    </Slider.Root>
  );
}
