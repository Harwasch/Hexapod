import { Columns2, Layers, List, Plus, Search } from "lucide-react";
import { useMemo, useState } from "react";

import { LAYER_CATEGORIES, type Layer, type LayerCategory } from "@twin/contracts";
import {
  EmptyState,
  GlassBadge,
  GlassButton,
  GlassInput,
  GlassSegmentedControl,
  Spinner,
} from "@twin/ui";

import { useLayers as useLayerCatalog } from "@/api/queries";
import { categoryLabel } from "@/lib/format";
import { useUi, type LayersMode } from "@/state/ui";

import { CompareControls } from "../compare/Compare";
import { FloatingPanel } from "../shell/FloatingPanel";
import { LayerCard } from "./LayerCard";
import { LayerFavourites } from "./LayerFavourites";

const CATEGORY_ORDER: LayerCategory[] = [...LAYER_CATEGORIES];

const MODES: { value: LayersMode; label: string; icon: typeof Layers }[] = [
  { value: "browse", label: "All layers", icon: List },
  { value: "compare", label: "Compare", icon: Columns2 },
];

/**
 * Layers: the favourites (imagery, vegetation, zones, tracks) at the top, then either the
 * whole catalog or Compare, a swipe between two of its layers.
 */
export function LayersPanel() {
  const open = useUi((s) => s.activePanel === "layers");
  const setPanel = useUi((s) => s.setPanel);
  const mode = useUi((s) => s.layersMode);
  const setMode = useUi((s) => s.setLayersMode);
  const openAdd = useUi((s) => s.openAdd);
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
          onClick={() => openAdd("link")}
          leadingIcon={<Plus size={14} aria-hidden="true" />}
          aria-label="Add a layer"
        >
          Add
        </GlassButton>
      }
    >
      <div className="glass-stack">
        <LayerFavourites />
        <GlassSegmentedControl
          aria-label="Layers mode"
          data-testid="layers-mode"
          block
          value={mode}
          onValueChange={setMode}
          options={MODES.map(({ value, label, icon: Icon }) => ({
            value,
            label,
            icon: <Icon size={13} aria-hidden="true" />,
          }))}
        />
        {mode === "compare" ? (
          <CompareControls />
        ) : (
          <LayerCatalog
            filter={filter}
            onFilter={setFilter}
            groups={groups}
            catalog={catalog.data ?? []}
            loading={catalog.isLoading}
            builtin={catalog.builtin}
            onAdd={() => openAdd("link")}
          />
        )}
      </div>
    </FloatingPanel>
  );
}

function LayerCatalog({
  filter,
  onFilter,
  groups,
  catalog,
  loading,
  builtin,
  onAdd,
}: {
  filter: string;
  onFilter: (value: string) => void;
  groups: { category: LayerCategory; layers: Layer[] }[];
  /** Every layer, filtered or not: a switch's undo needs its exclusive group's others. */
  catalog: readonly Layer[];
  loading: boolean;
  builtin: boolean;
  onAdd: () => void;
}) {
  return (
    <>
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
          onChange={(e) => onFilter(e.target.value)}
          style={{ paddingLeft: "2rem" }}
          data-testid="layers-filter"
        />
      </div>
      {builtin && <GlassBadge tone="warning">Built-in catalog (API offline)</GlassBadge>}
      {loading && (
        <div className="glass-row" style={{ justifyContent: "center", padding: "1rem" }}>
          <Spinner label="Loading catalog" />
        </div>
      )}
      {!loading && groups.length === 0 && (
        <EmptyState
          icon={<Layers size={28} />}
          title="No layers match"
          body="Try another term, or add your own data source."
          action={
            <GlassButton size="sm" onClick={onAdd}>
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
              <LayerCard key={layer.id} layer={layer} catalog={catalog} />
            ))}
          </ul>
        </section>
      ))}
    </>
  );
}
