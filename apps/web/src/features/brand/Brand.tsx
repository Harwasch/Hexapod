import { GlassPanel } from "@twin/ui";

export function Brand() {
  return (
    <GlassPanel pill soft className="brand" aria-label="Living World">
      <span className="brand__mark" aria-hidden="true" />
      <span className="brand__name">Living World</span>
      <span className="brand__tag">digital twin</span>
    </GlassPanel>
  );
}
