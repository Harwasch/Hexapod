import { MapPin, Plus } from "lucide-react";

import { EmptyState, GlassBadge, GlassButton, Spinner } from "@twin/ui";

import { useSites as useSiteCatalog } from "@/api/queries";
import { useScene } from "@/cesium/SceneContext";
import { useUi } from "@/state/ui";

import { FloatingPanel } from "../shell/FloatingPanel";
import { SiteCard } from "./SiteCard";

export function SitesPanel() {
  const open = useUi((s) => s.activePanel === "sites");
  const setPanel = useUi((s) => s.setPanel);
  const setAddDataOpen = useUi((s) => s.setAddDataOpen);
  const scene = useScene();
  const catalog = useSiteCatalog();
  return (
    <FloatingPanel
      open={open}
      title="Sites"
      onClose={() => setPanel(null)}
      testId="sites-panel"
      actions={
        <GlassButton
          size="sm"
          variant="ghost"
          onClick={() => setAddDataOpen(true)}
          leadingIcon={<Plus size={14} aria-hidden="true" />}
        >
          Add
        </GlassButton>
      }
    >
      <div className="glass-stack">
        {catalog.builtin && <GlassBadge tone="warning">Built-in demo (API offline)</GlassBadge>}
        {catalog.isLoading && (
          <div className="glass-row" style={{ justifyContent: "center", padding: "1rem" }}>
            <Spinner label="Loading sites" />
          </div>
        )}
        {!catalog.isLoading && (catalog.data?.length ?? 0) === 0 && (
          <EmptyState
            icon={<MapPin size={28} />}
            title="No sites yet"
            body="Register a reality model to see it embedded in the world."
            action={
              <GlassButton size="sm" onClick={() => setAddDataOpen(true)}>
                Add a site
              </GlassButton>
            }
          />
        )}
        <ul className="glass-list">
          {(catalog.data ?? []).map((site) => (
            <SiteCard key={site.id} site={site} onSelect={() => void scene?.sites.flyTo(site.id)} />
          ))}
        </ul>
      </div>
    </FloatingPanel>
  );
}
