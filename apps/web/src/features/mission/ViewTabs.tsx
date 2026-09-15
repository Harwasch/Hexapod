import { useMission, type MissionView } from "@/state/mission";

const TABS: { id: MissionView; label: string }[] = [
  { id: "map", label: "Map" },
  { id: "plan", label: "Plan" },
  { id: "fleet", label: "Fleet" },
];

/** Map / Plan / Fleet view switch (design: VIEW TABS). */
export function ViewTabs() {
  const view = useMission((s) => s.view);
  const setView = useMission((s) => s.setView);
  return (
    <div className="glass mc-tabs" role="tablist" aria-label="View" data-testid="view-tabs">
      {TABS.map((tab) => (
        <button
          key={tab.id}
          type="button"
          role="tab"
          aria-selected={view === tab.id}
          className={`mc-tabs__tab ${view === tab.id ? "is-on" : ""}`}
          onClick={() => setView(tab.id)}
          data-testid={`view-tab-${tab.id}`}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
