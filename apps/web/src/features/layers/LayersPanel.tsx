import { Layers, Plus, Search } from "lucide-react";
import { useMemo, useState } from "react";

import { LAYER_CATEGORIES, type Layer, type LayerCategory } from "@twin/contracts";
import { EmptyState, GlassBadge, GlassButton, GlassInput, Spinner } from "@twin/ui";

import { useLayers as useLayerCatalog } from "@/api/queries";
import { categoryLabel } from "@/lib/format";
import { useUi } from "@/state/ui";

import { FloatingPanel } from "../shell/FloatingPanel";
import { LayerCard } from "./LayerCard";

const CATEGORY_ORDER: LayerCategory[] = [...LAYER_CATEGORIES];

export function LayersPanel() {
  const open = useUi((s) => s.activePanel === "layers");
  const setPanel = useUi((s) => s.setPanel);
  const setAddDataOpen = useUi((s) => s.setAddDataOpen);
  const catalog = useLayerCatalog();
  const [filter, setFilter] = useState("");

  const groups = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    const byCategory = new Map<LayerCategory, Layer[]>();
    for (const layer of catalog.data ?? []) {
      const haystack =
        `${layer.name} ${layer.description ?? ""} ${layer.category} ${layer.coverage ?? ""}`.toLowerCase();
      if (needle && !haystack.includes(needle)) continue;
      const list = byCategory.get(layer.category) ?? [];
      list.push(layer);
      byCategory.set(layer.category, list);
    }
    return CATEGORY_ORDER.filter((c) => byCategory.has(c)).map((c) => ({
      category: c,
      layers: byCategory.get(c) ?? [],
    }));
  }, [catalog.data, filter]);

  return (
    <FloatingPanel
      open={open}
      title="Layers"
      onClose={() => setPanel(null)}
      testId="layers-panel"
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
        <div style={{ position: "relative" }}>
          <Search
            size={14}
            className="glass-subtle"
            aria-hidden="true"
            style={{ position: "absolute", left: 10, top: 11 }}
          />
          <GlassInput
            aria-label="Filter layers"
            placeholder="Filter layers…"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            style={{ paddingLeft: "2rem" }}
            data-testid="layers-filter"
          />
        </div>
        {catalog.builtin && <GlassBadge tone="warning">Built-in catalog (API offline)</GlassBadge>}
        {catalog.isLoading && (
          <div className="glass-row" style={{ justifyContent: "center", padding: "1rem" }}>
            <Spinner label="Loading catalog" />
          </div>
        )}
        {!catalog.isLoading && groups.length === 0 && (
          <EmptyState
            icon={<Layers size={28} />}
            title="No layers match"
            body="Try another term, or add your own data source."
            action={
              <GlassButton size="sm" onClick={() => setAddDataOpen(true)}>
                Add data
              </GlassButton>
            }
          />
        )}
        {groups.map(({ category, layers }) => (
          <section key={category} aria-label={categoryLabel(category)}>
            <div className="category">
              <p className="glass-eyebrow">{categoryLabel(category)}</p>
              <span className="glass-subtle" style={{ fontSize: "var(--text-xs)" }}>
                {layers.length}
              </span>
            </div>
            <ul className="glass-list">
              {layers.map((layer) => (
                <LayerCard key={layer.id} layer={layer} />
              ))}
            </ul>
          </section>
        ))}
      </div>
    </FloatingPanel>
  );
}
