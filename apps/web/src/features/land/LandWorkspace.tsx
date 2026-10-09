import { useId, type KeyboardEvent } from "react";
import type { LandArea } from "@twin/contracts";
import { useLandContext } from "@/state/landContext";
import { LandSavedViews } from "./LandSavedViews";
import { LandResearch } from "./LandResearch";
import { LandInventory } from "./LandInventory";
import { LandEcology } from "./LandEcology";
import { LandScenarios } from "./LandScenarios";
import { LandActions } from "./LandActions";
import { LandDocuments } from "./LandDocuments";

const sections = [
  ["discover", "Discover"],
  ["records", "Records"],
  ["inventory", "Assets"],
  ["ecology", "Ecology"],
  ["scenarios", "Scenarios"],
  ["actions", "Actions"],
] as const;
export function LandWorkspace({ land }: { land: LandArea }) {
  const id = useId();
  const section = useLandContext((state) => state.section);
  const change = useLandContext((state) => state.setSection);
  const key = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    let next: number;
    if (event.key === "ArrowRight") next = (index + 1) % sections.length;
    else if (event.key === "ArrowLeft") next = (index + sections.length - 1) % sections.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = sections.length - 1;
    else return;
    event.preventDefault();
    const chosen = sections[next];
    if (!chosen) return;
    change(chosen[0]);
    document.getElementById(`${id}-${chosen[0]}-tab`)?.focus();
  };
  return (
    <div className="land-exploration-workspace">
      <LandSavedViews land={land} />
      <div className="land-workspace-tabs" role="tablist" aria-label="Land workspace">
        {sections.map(([value, label], index) => (
          <button
            key={value}
            type="button"
            id={`${id}-${value}-tab`}
            role="tab"
            tabIndex={section === value ? 0 : -1}
            aria-selected={section === value}
            aria-controls={`${id}-${value}-content`}
            onKeyDown={(event) => key(event, index)}
            onClick={() => change(value)}
          >
            {label}
          </button>
        ))}
      </div>
      {sections.map(([value]) => (
        <div
          key={value}
          id={`${id}-${value}-content`}
          role="tabpanel"
          aria-labelledby={`${id}-${value}-tab`}
          hidden={section !== value}
          tabIndex={0}
        >
          {value === "discover" ? (
            <LandResearch land={land} />
          ) : value === "records" ? (
            <LandDocuments land={land} />
          ) : value === "inventory" ? (
            <LandInventory land={land} />
          ) : value === "ecology" ? (
            <LandEcology land={land} />
          ) : value === "scenarios" ? (
            <LandScenarios land={land} />
          ) : (
            <LandActions land={land} />
          )}
        </div>
      ))}
    </div>
  );
}
