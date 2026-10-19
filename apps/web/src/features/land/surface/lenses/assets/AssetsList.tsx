import { useMemo, useState } from "react";
import { Search } from "lucide-react";

import { GlassInput } from "@twin/ui";

import { MapLink } from "../../mapLinks";
import { useFlyTo } from "../../mapHooks";
import {
  CATEGORIES,
  CATEGORY_ORDER,
  CONDITIONS,
  needsAttention,
  STATUSES,
  type Asset,
  type AssetCategory,
  type Inspection,
} from "./assetsModel";

type Filter = "all" | "attention" | "retired" | AssetCategory;

const FIRST = 6;
const PAGE = 50;

/**
 * Every asset as one short row: a dot in its map colour, its name and kind, and its condition
 * when it needs a look. Hovering a row points at it on the map; choosing it selects it.
 * Kind chips filter the list and double as the map's legend.
 */
export function AssetsList({
  assets,
  latest,
  colorOf,
  selectedId,
  onSelect,
}: {
  assets: readonly Asset[];
  latest: Readonly<Record<string, Inspection | null | undefined>>;
  colorOf: (asset: Asset) => string;
  selectedId: string | null;
  onSelect: (asset: Asset) => void;
}) {
  const fly = useFlyTo();
  // With only retired assets left, show those rather than an empty list.
  const [filter, setFilter] = useState<Filter>(() =>
    assets.some((asset) => asset.status !== "retired") ? "all" : "retired",
  );
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(FIRST);
  const active = useMemo(() => assets.filter((asset) => asset.status !== "retired"), [assets]);
  const retired = assets.length - active.length;
  const attention = active.filter((asset) => needsAttention(latest[asset.id]?.condition)).length;
  const counts = new Map<AssetCategory, number>();
  for (const asset of active) counts.set(asset.category, (counts.get(asset.category) ?? 0) + 1);

  const shown = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const pool =
      filter === "retired" ? assets.filter((asset) => asset.status === "retired") : active;
    return pool
      .filter(
        (asset) =>
          (filter === "all" ||
            filter === "retired" ||
            (filter === "attention"
              ? needsAttention(latest[asset.id]?.condition)
              : asset.category === filter)) &&
          (!needle || asset.name.toLowerCase().includes(needle)),
      )
      .sort((a, b) => {
        const rank = (asset: Asset) => {
          const condition = latest[asset.id]?.condition;
          return condition === "critical" ? 0 : condition === "poor" ? 1 : 2;
        };
        return rank(a) - rank(b) || a.name.localeCompare(b.name);
      });
  }, [assets, active, filter, query, latest]);
  const visible = shown.slice(0, limit);
  const next = limit === FIRST ? PAGE : limit + PAGE;

  const chip = (value: Filter, label: string, count: number, color?: string) => (
    <button
      key={value}
      type="button"
      className="assets-filter"
      aria-pressed={filter === value}
      onClick={() => {
        setFilter(filter === value ? "all" : value);
        setLimit(FIRST);
      }}
    >
      {color && <i style={{ background: color }} aria-hidden="true" />}
      {label} <b>{count}</b>
    </button>
  );

  return (
    <div className="assets-browse">
      {assets.length > 8 && (
        <div className="assets-search">
          <Search size={14} aria-hidden="true" />
          <GlassInput
            type="search"
            aria-label="Find an asset"
            placeholder="Find an asset"
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              setLimit(FIRST);
            }}
          />
        </div>
      )}
      <div className="assets-filters" role="group" aria-label="Show">
        {chip("all", "All", active.length)}
        {CATEGORY_ORDER.filter((category) => counts.has(category)).map((category) =>
          chip(
            category,
            CATEGORIES[category].label,
            counts.get(category) ?? 0,
            CATEGORIES[category].color,
          ),
        )}
        {attention > 0 && chip("attention", "Needs attention", attention, CONDITIONS.poor.color)}
        {retired > 0 && chip("retired", "Retired", retired)}
      </div>
      <ul className="assets-list" aria-label="Assets">
        {visible.map((asset) => {
          const condition = latest[asset.id]?.condition;
          const flag = needsAttention(condition);
          return (
            <MapLink
              key={asset.id}
              as="li"
              fly={false}
              tone={flag ? "warn" : "accent"}
              className="assets-row"
              target={{ key: `asset:${asset.id}`, geometry: asset.geometry }}
            >
              <button
                type="button"
                className="assets-row__main"
                aria-current={selectedId === asset.id || undefined}
                onClick={() => {
                  onSelect(asset);
                  fly(asset.geometry);
                }}
              >
                <i
                  className="assets-dot"
                  style={{ background: colorOf(asset) }}
                  aria-hidden="true"
                />
                <span className="assets-row__text">
                  <span className="assets-row__name">{asset.name}</span>
                  <span className="assets-row__sub">
                    {CATEGORIES[asset.category].one}
                    {asset.status === "candidate" && ` · ${STATUSES.candidate.toLowerCase()}`}
                    {!asset.intersectsLand && " · outside your land"}
                  </span>
                </span>
                {flag && condition && (
                  <span className="assets-flag">{CONDITIONS[condition].label}</span>
                )}
              </button>
            </MapLink>
          );
        })}
      </ul>
      {!shown.length && <p className="land-muted">Nothing matches.</p>}
      {shown.length > limit ? (
        <button type="button" className="land-link" onClick={() => setLimit(next)}>
          {next >= shown.length ? `Show all ${shown.length}` : `Show ${next - limit} more`}
        </button>
      ) : (
        limit > FIRST && (
          <button type="button" className="land-link" onClick={() => setLimit(FIRST)}>
            Show fewer
          </button>
        )
      )}
    </div>
  );
}
